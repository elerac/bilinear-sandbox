#include "kernels/demosaicing_fast.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define BILINEAR_HAS_NEON 1
#else
#define BILINEAR_HAS_NEON 0
#endif

namespace bilinear {
namespace {

template <typename T>
T avg2(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        return (a + b) / T{2};
    } else {
        return static_cast<T>((static_cast<std::uint32_t>(a) + static_cast<std::uint32_t>(b) + 1) / 2);
    }
}

template <typename T>
T avg4(T a, T b, T c, T d) {
    if constexpr (std::is_floating_point_v<T>) {
        return (a + b + c + d) / T{4};
    } else {
        return static_cast<T>(
            (
                static_cast<std::uint32_t>(a) +
                static_cast<std::uint32_t>(b) +
                static_cast<std::uint32_t>(c) +
                static_cast<std::uint32_t>(d) +
                2
            ) / 4
        );
    }
}

constexpr std::size_t kParallelMinPixels = 1024 * 1024;
constexpr std::size_t kParallelMaxWorkers = 8;

std::size_t red_blue_pair_count(std::size_t first_y, std::size_t height) {
    if (first_y + 2 >= height) {
        return 0;
    }
    return ((height - 3) - first_y) / 2 + 1;
}

std::size_t choose_parallel_workers(std::size_t pair_count, std::size_t height, std::size_t width) {
    if (pair_count < 16 || height * width < kParallelMinPixels) {
        return 1;
    }

    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads <= 1) {
        return 1;
    }

    return std::min<std::size_t>({pair_count, kParallelMaxWorkers, static_cast<std::size_t>(hardware_threads)});
}

class ParallelRowPool {
public:
    explicit ParallelRowPool(std::size_t helper_count) {
        threads_.reserve(helper_count);
        for (std::size_t index = 0; index < helper_count; ++index) {
            threads_.emplace_back([this, index]() {
                worker_loop(index);
            });
        }
    }

    ~ParallelRowPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        start_cv_.notify_all();
        for (std::thread &thread : threads_) {
            thread.join();
        }
    }

    std::size_t helper_count() const {
        return threads_.size();
    }

    template <typename Work>
    void run_chunks(std::size_t count, std::size_t worker_count, Work work) {
        const std::lock_guard<std::mutex> run_lock(run_mutex_);
        const std::size_t active_helpers = std::min(worker_count - 1, threads_.size());
        const std::size_t active_workers = active_helpers + 1;
        const std::size_t chunk_size = (count + active_workers - 1) / active_workers;

        auto run_chunk = [&](std::size_t worker_index) {
            const std::size_t begin = worker_index * chunk_size;
            const std::size_t end = std::min(count, begin + chunk_size);
            if (begin < end) {
                work(begin, end);
            }
        };

        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_helpers_ = active_helpers;
            completed_helpers_ = 0;
            job_ = run_chunk;
            ++generation_;
        }
        start_cv_.notify_all();

        run_chunk(0);

        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [&]() {
                return completed_helpers_ == active_helpers_;
            });
            job_ = nullptr;
            active_helpers_ = 0;
        }
    }

private:
    void worker_loop(std::size_t helper_index) {
        std::uint64_t seen_generation = 0;

        while (true) {
            std::function<void(std::size_t)> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_cv_.wait(lock, [&]() {
                    return stop_ || seen_generation != generation_;
                });
                if (stop_) {
                    return;
                }

                seen_generation = generation_;
                if (helper_index >= active_helpers_) {
                    continue;
                }
                job = job_;
            }

            job(helper_index + 1);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++completed_helpers_;
                if (completed_helpers_ == active_helpers_) {
                    done_cv_.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> threads_;
    std::mutex run_mutex_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::function<void(std::size_t)> job_;
    std::uint64_t generation_ = 0;
    std::size_t active_helpers_ = 0;
    std::size_t completed_helpers_ = 0;
    bool stop_ = false;
};

std::size_t max_thread_pool_workers() {
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads <= 1) {
        return 1;
    }
    return std::min<std::size_t>({kParallelMaxWorkers, static_cast<std::size_t>(hardware_threads)});
}

ParallelRowPool &parallel_row_pool() {
    static ParallelRowPool pool(max_thread_pool_workers() - 1);
    return pool;
}

template <typename Work>
void parallel_for_chunks(std::size_t count, std::size_t worker_count, Work work) {
    if (worker_count <= 1 || count <= 1) {
        work(0, count);
        return;
    }

    ParallelRowPool &pool = parallel_row_pool();
    if (pool.helper_count() == 0) {
        work(0, count);
        return;
    }

    pool.run_chunks(count, worker_count, work);
}

#if BILINEAR_HAS_NEON
uint8x16_t avg4_u8x16(uint8x16_t a, uint8x16_t b, uint8x16_t c, uint8x16_t d) {
    uint16x8_t low = vaddq_u16(
        vaddl_u8(vget_low_u8(a), vget_low_u8(b)),
        vaddl_u8(vget_low_u8(c), vget_low_u8(d))
    );
    low = vaddq_u16(low, vdupq_n_u16(2));

    uint16x8_t high = vaddq_u16(
        vaddl_u8(vget_high_u8(a), vget_high_u8(b)),
        vaddl_u8(vget_high_u8(c), vget_high_u8(d))
    );
    high = vaddq_u16(high, vdupq_n_u16(2));

    return vcombine_u8(vshrn_n_u16(low, 2), vshrn_n_u16(high, 2));
}

uint16x8_t avg4_u16x8(uint16x8_t a, uint16x8_t b, uint16x8_t c, uint16x8_t d) {
    uint32x4_t low = vaddq_u32(
        vaddl_u16(vget_low_u16(a), vget_low_u16(b)),
        vaddl_u16(vget_low_u16(c), vget_low_u16(d))
    );
    low = vaddq_u32(low, vdupq_n_u32(2));

    uint32x4_t high = vaddq_u32(
        vaddl_u16(vget_high_u16(a), vget_high_u16(b)),
        vaddl_u16(vget_high_u16(c), vget_high_u16(d))
    );
    high = vaddq_u32(high, vdupq_n_u32(2));

    return vcombine_u16(vshrn_n_u32(low, 2), vshrn_n_u32(high, 2));
}

uint8x16_t even_lane_mask_u8() {
    static constexpr std::uint8_t mask[16] = {
        0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0,
        0xff, 0, 0xff, 0, 0xff, 0, 0xff, 0,
    };
    return vld1q_u8(mask);
}

uint8x16_t odd_lane_mask_u8() {
    static constexpr std::uint8_t mask[16] = {
        0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff,
        0, 0xff, 0, 0xff, 0, 0xff, 0, 0xff,
    };
    return vld1q_u8(mask);
}

uint16x8_t even_lane_mask_u16() {
    static constexpr std::uint16_t mask[8] = {
        0xffff, 0, 0xffff, 0, 0xffff, 0, 0xffff, 0,
    };
    return vld1q_u16(mask);
}

uint16x8_t odd_lane_mask_u16() {
    static constexpr std::uint16_t mask[8] = {
        0, 0xffff, 0, 0xffff, 0, 0xffff, 0, 0xffff,
    };
    return vld1q_u16(mask);
}
#endif

template <typename T>
void write_red_site(const T *prev, const T *curr, const T *next, T *out, std::size_t x) {
    out[2] = curr[x];
    out[1] = static_cast<T>(avg4(prev[x], next[x], curr[x - 1], curr[x + 1]));
    out[0] = static_cast<T>(avg4(prev[x - 1], prev[x + 1], next[x - 1], next[x + 1]));
}

template <typename T>
void write_blue_site(const T *prev, const T *curr, const T *next, T *out, std::size_t x) {
    out[0] = curr[x];
    out[1] = static_cast<T>(avg4(prev[x], next[x], curr[x - 1], curr[x + 1]));
    out[2] = static_cast<T>(avg4(prev[x - 1], prev[x + 1], next[x - 1], next[x + 1]));
}

template <typename T>
void write_green_horizontal_red(const T *prev, const T *curr, const T *next, T *out, std::size_t x) {
    out[1] = curr[x];
    out[2] = static_cast<T>(avg2(curr[x - 1], curr[x + 1]));
    out[0] = static_cast<T>(avg2(prev[x], next[x]));
}

template <typename T>
void write_green_vertical_red(const T *prev, const T *curr, const T *next, T *out, std::size_t x) {
    out[1] = curr[x];
    out[0] = static_cast<T>(avg2(curr[x - 1], curr[x + 1]));
    out[2] = static_cast<T>(avg2(prev[x], next[x]));
}

template <typename T, bool FirstIsRed>
void write_red_row(const T *prev, const T *curr, const T *next, T *row_out, std::size_t width) {
    const std::size_t end = width - 1;
    std::size_t x = 1;

    for (; x + 1 < end; x += 2) {
        T *out = row_out + x * 3;
        if constexpr (FirstIsRed) {
            write_red_site(prev, curr, next, out, x);
            write_green_horizontal_red(prev, curr, next, out + 3, x + 1);
        } else {
            write_green_horizontal_red(prev, curr, next, out, x);
            write_red_site(prev, curr, next, out + 3, x + 1);
        }
    }

    if (x < end) {
        T *out = row_out + x * 3;
        if constexpr (FirstIsRed) {
            write_red_site(prev, curr, next, out, x);
        } else {
            write_green_horizontal_red(prev, curr, next, out, x);
        }
    }
}

template <typename T, bool FirstIsBlue>
void write_blue_row(const T *prev, const T *curr, const T *next, T *row_out, std::size_t width) {
    const std::size_t end = width - 1;
    std::size_t x = 1;

    for (; x + 1 < end; x += 2) {
        T *out = row_out + x * 3;
        if constexpr (FirstIsBlue) {
            write_blue_site(prev, curr, next, out, x);
            write_green_vertical_red(prev, curr, next, out + 3, x + 1);
        } else {
            write_green_vertical_red(prev, curr, next, out, x);
            write_blue_site(prev, curr, next, out + 3, x + 1);
        }
    }

    if (x < end) {
        T *out = row_out + x * 3;
        if constexpr (FirstIsBlue) {
            write_blue_site(prev, curr, next, out, x);
        } else {
            write_green_vertical_red(prev, curr, next, out, x);
        }
    }
}

template <typename T, bool RedX>
void write_red_blue_row_pair(
    const T *red_prev,
    const T *red_row,
    const T *blue_row,
    const T *blue_next,
    T *red_out,
    T *blue_out,
    std::size_t width
) {
    // Process one 2x2 Bayer tile per iteration while preserving exact border and averaging behavior.
    const std::size_t end = width - 1;
    std::size_t x = 1;
    const T *red_prev_tile = red_prev;
    const T *red_row_tile = red_row;
    const T *blue_row_tile = blue_row;
    const T *blue_next_tile = blue_next;
    T *red_pair = red_out + x * 3;
    T *blue_pair = blue_out + x * 3;

#if BILINEAR_HAS_NEON
    if constexpr (std::is_same_v<T, std::uint8_t>) {
        const uint8x16_t red_site_mask = RedX ? even_lane_mask_u8() : odd_lane_mask_u8();
        const uint8x16_t blue_site_mask = RedX ? odd_lane_mask_u8() : even_lane_mask_u8();

        for (; x + 15 < end; x += 16) {
            const uint8x16_t red_prev_left = vld1q_u8(red_prev_tile);
            const uint8x16_t red_prev_center = vld1q_u8(red_prev_tile + 1);
            const uint8x16_t red_prev_right = vld1q_u8(red_prev_tile + 2);

            const uint8x16_t red_row_left = vld1q_u8(red_row_tile);
            const uint8x16_t red_row_center = vld1q_u8(red_row_tile + 1);
            const uint8x16_t red_row_right = vld1q_u8(red_row_tile + 2);

            const uint8x16_t blue_row_left = vld1q_u8(blue_row_tile);
            const uint8x16_t blue_row_center = vld1q_u8(blue_row_tile + 1);
            const uint8x16_t blue_row_right = vld1q_u8(blue_row_tile + 2);

            const uint8x16_t blue_next_left = vld1q_u8(blue_next_tile);
            const uint8x16_t blue_next_center = vld1q_u8(blue_next_tile + 1);
            const uint8x16_t blue_next_right = vld1q_u8(blue_next_tile + 2);

            const uint8x16_t red_site_b = avg4_u8x16(
                red_prev_left,
                red_prev_right,
                blue_row_left,
                blue_row_right
            );
            const uint8x16_t red_site_g = avg4_u8x16(
                red_prev_center,
                blue_row_center,
                red_row_left,
                red_row_right
            );
            const uint8x16_t red_site_r = red_row_center;

            const uint8x16_t red_green_b = vrhaddq_u8(red_prev_center, blue_row_center);
            const uint8x16_t red_green_g = red_row_center;
            const uint8x16_t red_green_r = vrhaddq_u8(red_row_left, red_row_right);

            uint8x16x3_t red_pixels;
            red_pixels.val[0] = vbslq_u8(red_site_mask, red_site_b, red_green_b);
            red_pixels.val[1] = vbslq_u8(red_site_mask, red_site_g, red_green_g);
            red_pixels.val[2] = vbslq_u8(red_site_mask, red_site_r, red_green_r);
            vst3q_u8(red_pair, red_pixels);

            const uint8x16_t blue_green_b = vrhaddq_u8(blue_row_left, blue_row_right);
            const uint8x16_t blue_green_g = blue_row_center;
            const uint8x16_t blue_green_r = vrhaddq_u8(red_row_center, blue_next_center);

            const uint8x16_t blue_site_b = blue_row_center;
            const uint8x16_t blue_site_g = avg4_u8x16(
                red_row_center,
                blue_next_center,
                blue_row_left,
                blue_row_right
            );
            const uint8x16_t blue_site_r = avg4_u8x16(
                red_row_left,
                red_row_right,
                blue_next_left,
                blue_next_right
            );

            uint8x16x3_t blue_pixels;
            blue_pixels.val[0] = vbslq_u8(blue_site_mask, blue_site_b, blue_green_b);
            blue_pixels.val[1] = vbslq_u8(blue_site_mask, blue_site_g, blue_green_g);
            blue_pixels.val[2] = vbslq_u8(blue_site_mask, blue_site_r, blue_green_r);
            vst3q_u8(blue_pair, blue_pixels);

            red_prev_tile += 16;
            red_row_tile += 16;
            blue_row_tile += 16;
            blue_next_tile += 16;
            red_pair += 48;
            blue_pair += 48;
        }
    }

    if constexpr (std::is_same_v<T, std::uint16_t>) {
        const uint16x8_t red_site_mask = RedX ? even_lane_mask_u16() : odd_lane_mask_u16();
        const uint16x8_t blue_site_mask = RedX ? odd_lane_mask_u16() : even_lane_mask_u16();

        for (; x + 7 < end; x += 8) {
            const uint16x8_t red_prev_left = vld1q_u16(red_prev_tile);
            const uint16x8_t red_prev_center = vld1q_u16(red_prev_tile + 1);
            const uint16x8_t red_prev_right = vld1q_u16(red_prev_tile + 2);

            const uint16x8_t red_row_left = vld1q_u16(red_row_tile);
            const uint16x8_t red_row_center = vld1q_u16(red_row_tile + 1);
            const uint16x8_t red_row_right = vld1q_u16(red_row_tile + 2);

            const uint16x8_t blue_row_left = vld1q_u16(blue_row_tile);
            const uint16x8_t blue_row_center = vld1q_u16(blue_row_tile + 1);
            const uint16x8_t blue_row_right = vld1q_u16(blue_row_tile + 2);

            const uint16x8_t blue_next_left = vld1q_u16(blue_next_tile);
            const uint16x8_t blue_next_center = vld1q_u16(blue_next_tile + 1);
            const uint16x8_t blue_next_right = vld1q_u16(blue_next_tile + 2);

            const uint16x8_t red_site_b = avg4_u16x8(
                red_prev_left,
                red_prev_right,
                blue_row_left,
                blue_row_right
            );
            const uint16x8_t red_site_g = avg4_u16x8(
                red_prev_center,
                blue_row_center,
                red_row_left,
                red_row_right
            );
            const uint16x8_t red_site_r = red_row_center;

            const uint16x8_t red_green_b = vrhaddq_u16(red_prev_center, blue_row_center);
            const uint16x8_t red_green_g = red_row_center;
            const uint16x8_t red_green_r = vrhaddq_u16(red_row_left, red_row_right);

            uint16x8x3_t red_pixels;
            red_pixels.val[0] = vbslq_u16(red_site_mask, red_site_b, red_green_b);
            red_pixels.val[1] = vbslq_u16(red_site_mask, red_site_g, red_green_g);
            red_pixels.val[2] = vbslq_u16(red_site_mask, red_site_r, red_green_r);
            vst3q_u16(red_pair, red_pixels);

            const uint16x8_t blue_green_b = vrhaddq_u16(blue_row_left, blue_row_right);
            const uint16x8_t blue_green_g = blue_row_center;
            const uint16x8_t blue_green_r = vrhaddq_u16(red_row_center, blue_next_center);

            const uint16x8_t blue_site_b = blue_row_center;
            const uint16x8_t blue_site_g = avg4_u16x8(
                red_row_center,
                blue_next_center,
                blue_row_left,
                blue_row_right
            );
            const uint16x8_t blue_site_r = avg4_u16x8(
                red_row_left,
                red_row_right,
                blue_next_left,
                blue_next_right
            );

            uint16x8x3_t blue_pixels;
            blue_pixels.val[0] = vbslq_u16(blue_site_mask, blue_site_b, blue_green_b);
            blue_pixels.val[1] = vbslq_u16(blue_site_mask, blue_site_g, blue_green_g);
            blue_pixels.val[2] = vbslq_u16(blue_site_mask, blue_site_r, blue_green_r);
            vst3q_u16(blue_pair, blue_pixels);

            red_prev_tile += 8;
            red_row_tile += 8;
            blue_row_tile += 8;
            blue_next_tile += 8;
            red_pair += 24;
            blue_pair += 24;
        }
    }
#endif

    for (; x + 1 < end; x += 2) {
        const T red_row_xm1 = red_row_tile[0];
        const T red_row_x = red_row_tile[1];
        const T red_row_xp1 = red_row_tile[2];
        const T red_row_xp2 = red_row_tile[3];

        const T blue_row_xm1 = blue_row_tile[0];
        const T blue_row_x = blue_row_tile[1];
        const T blue_row_xp1 = blue_row_tile[2];
        const T blue_row_xp2 = blue_row_tile[3];

        if constexpr (RedX) {
            const T red_prev_xm1 = red_prev_tile[0];
            const T red_prev_x = red_prev_tile[1];
            const T red_prev_xp1 = red_prev_tile[2];
            const T blue_next_x = blue_next_tile[1];
            const T blue_next_xp1 = blue_next_tile[2];
            const T blue_next_xp2 = blue_next_tile[3];

            red_pair[0] = static_cast<T>(avg4(red_prev_xm1, red_prev_xp1, blue_row_xm1, blue_row_xp1));
            red_pair[1] = static_cast<T>(avg4(red_prev_x, blue_row_x, red_row_xm1, red_row_xp1));
            red_pair[2] = static_cast<T>(red_row_x);

            red_pair[3] = static_cast<T>(avg2(red_prev_xp1, blue_row_xp1));
            red_pair[4] = static_cast<T>(red_row_xp1);
            red_pair[5] = static_cast<T>(avg2(red_row_x, red_row_xp2));

            blue_pair[0] = static_cast<T>(avg2(blue_row_xm1, blue_row_xp1));
            blue_pair[1] = static_cast<T>(blue_row_x);
            blue_pair[2] = static_cast<T>(avg2(red_row_x, blue_next_x));

            blue_pair[3] = static_cast<T>(blue_row_xp1);
            blue_pair[4] = static_cast<T>(avg4(red_row_xp1, blue_next_xp1, blue_row_x, blue_row_xp2));
            blue_pair[5] = static_cast<T>(avg4(red_row_x, red_row_xp2, blue_next_x, blue_next_xp2));
        } else {
            const T red_prev_x = red_prev_tile[1];
            const T red_prev_xp1 = red_prev_tile[2];
            const T red_prev_xp2 = red_prev_tile[3];
            const T blue_next_xm1 = blue_next_tile[0];
            const T blue_next_x = blue_next_tile[1];
            const T blue_next_xp1 = blue_next_tile[2];

            red_pair[0] = static_cast<T>(avg2(red_prev_x, blue_row_x));
            red_pair[1] = static_cast<T>(red_row_x);
            red_pair[2] = static_cast<T>(avg2(red_row_xm1, red_row_xp1));

            red_pair[3] = static_cast<T>(avg4(red_prev_x, red_prev_xp2, blue_row_x, blue_row_xp2));
            red_pair[4] = static_cast<T>(avg4(red_prev_xp1, blue_row_xp1, red_row_x, red_row_xp2));
            red_pair[5] = static_cast<T>(red_row_xp1);

            blue_pair[0] = static_cast<T>(blue_row_x);
            blue_pair[1] = static_cast<T>(avg4(red_row_x, blue_next_x, blue_row_xm1, blue_row_xp1));
            blue_pair[2] = static_cast<T>(avg4(red_row_xm1, red_row_xp1, blue_next_xm1, blue_next_xp1));

            blue_pair[3] = static_cast<T>(avg2(blue_row_x, blue_row_xp2));
            blue_pair[4] = static_cast<T>(blue_row_xp1);
            blue_pair[5] = static_cast<T>(avg2(red_row_xp1, blue_next_xp1));
        }

        red_prev_tile += 2;
        red_row_tile += 2;
        blue_row_tile += 2;
        blue_next_tile += 2;
        red_pair += 6;
        blue_pair += 6;
    }

    if (x < end) {
        if constexpr (RedX) {
            write_red_site(red_prev, red_row, blue_row, red_pair, x);
            write_green_vertical_red(red_row, blue_row, blue_next, blue_pair, x);
        } else {
            write_green_horizontal_red(red_prev, red_row, blue_row, red_pair, x);
            write_blue_site(red_row, blue_row, blue_next, blue_pair, x);
        }
    }
}

template <typename T>
void copy_inner_border_fast(T *output, std::size_t height, std::size_t width) {
    const std::size_t row_items = width * 3;
    const std::size_t inner_items = (width - 2) * 3;
    std::copy_n(output + row_items + 3, inner_items, output + 3);
    std::copy_n(output + (height - 2) * row_items + 3, inner_items, output + (height - 1) * row_items + 3);

    for (std::size_t y = 0; y < height; ++y) {
        T *row = output + y * row_items;
        row[0] = row[3];
        row[1] = row[4];
        row[2] = row[5];

        T *right = row + (width - 1) * 3;
        const T *right_source = row + (width - 2) * 3;
        right[0] = right_source[0];
        right[1] = right_source[1];
        right[2] = right_source[2];
    }
}

template <typename T, bool RedY, bool RedX>
void demosaic_bgr_fast_pattern(const T *bayer, T *output, std::size_t height, std::size_t width) {
    if (height < 3 || width < 3) {
        std::fill_n(output, height * width * 3, T{});
        return;
    }

    const std::size_t row_items = width * 3;
    std::size_t y = 1;

    if constexpr (!RedY) {
        write_blue_row<T, !RedX>(
            bayer,
            bayer + width,
            bayer + 2 * width,
            output + row_items,
            width
        );
        ++y;
    }

    const std::size_t first_pair_y = y;
    const std::size_t pair_count = red_blue_pair_count(first_pair_y, height);
    const std::size_t worker_count = choose_parallel_workers(pair_count, height, width);

    parallel_for_chunks(pair_count, worker_count, [=](std::size_t begin, std::size_t end) {
        for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
            const std::size_t row_y = first_pair_y + pair_index * 2;
            const T *red_prev = bayer + (row_y - 1) * width;
            const T *red_row = bayer + row_y * width;
            const T *blue_row = bayer + (row_y + 1) * width;
            const T *blue_next = bayer + (row_y + 2) * width;
            T *red_out = output + row_y * row_items;
            T *blue_out = red_out + row_items;

            write_red_blue_row_pair<T, RedX>(red_prev, red_row, blue_row, blue_next, red_out, blue_out, width);
        }
    });
    y = first_pair_y + pair_count * 2;

    if (y < height - 1) {
        const T *prev = bayer + (y - 1) * width;
        const T *curr = bayer + y * width;
        const T *next = bayer + (y + 1) * width;
        T *row_out = output + y * row_items;

        write_red_row<T, RedX>(prev, curr, next, row_out, width);
    }

    copy_inner_border_fast(output, height, width);
}

template <typename T>
void demosaic_bgr_fast_impl(
    const T *bayer,
    T *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
) {
    switch (pattern) {
    case BayerPattern::RGGB:
        demosaic_bgr_fast_pattern<T, false, false>(bayer, output, height, width);
        break;
    case BayerPattern::GRBG:
        demosaic_bgr_fast_pattern<T, false, true>(bayer, output, height, width);
        break;
    case BayerPattern::BGGR:
        demosaic_bgr_fast_pattern<T, true, true>(bayer, output, height, width);
        break;
    case BayerPattern::GBRG:
        demosaic_bgr_fast_pattern<T, true, false>(bayer, output, height, width);
        break;
    }
}

}  // namespace

void demosaic_bgr_fast(
    const std::uint8_t *bayer,
    std::uint8_t *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
) {
    demosaic_bgr_fast_impl(bayer, output, height, width, pattern);
}

void demosaic_bgr_fast(
    const std::uint16_t *bayer,
    std::uint16_t *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
) {
    demosaic_bgr_fast_impl(bayer, output, height, width, pattern);
}

void demosaic_bgr_fast(
    const float *bayer,
    float *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
) {
    demosaic_bgr_fast_impl(bayer, output, height, width, pattern);
}

void demosaic_bgr_fast(
    const double *bayer,
    double *output,
    std::size_t height,
    std::size_t width,
    BayerPattern pattern
) {
    demosaic_bgr_fast_impl(bayer, output, height, width, pattern);
}

}  // namespace bilinear
