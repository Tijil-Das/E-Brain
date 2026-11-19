#include <cstddef>
#include <stdexcept>
#include <vector>
#include <algorithm>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t cap) : buf_(cap), cap_(cap) {}

    void push(T value) {
        if (full()) throw std::overflow_error("RingBuffer full");
        buf_[tail_] = std::move(value);
        tail_ = (tail_ + 1) % cap_;
        ++size_;
    }

    T pop() {
        if (empty()) throw std::underflow_error("RingBuffer empty");
        T v = std::move(buf_[head_]);
        head_ = (head_ + 1) % cap_;
        --size_;
        return v;
    }

    [[nodiscard]] bool   empty() const noexcept { return size_ == 0; }
    [[nodiscard]] bool   full()  const noexcept { return size_ == cap_; }
    [[nodiscard]] size_t size()  const noexcept { return size_; }

private:
    std::vector<T> buf_;
    std::size_t cap_, head_ = 0, tail_ = 0, size_ = 0;
};
