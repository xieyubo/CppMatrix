module;

#include <coroutine>
#include <future>
#include <mutex>

export module cnn:promise;

namespace cnn {

export template <typename T>
class PromiseState {
public:
    bool IsReady() const { return m_future.wait_for(std::chrono::milliseconds {}) == std::future_status::ready; }

    void SetValue(T t)
    {
        m_promise.set_value(std::move(t));
        if (auto h = std::move(m_h)) {
            h();
        }
    }

    T GetValue() { return std::move(m_future.get()); }

    void SetCoroutineHandle(std::coroutine_handle<> h)
    {
        if (IsReady()) {
            h();
        } else {
            m_h = std::move(h);
        }
    }

private:
    std::promise<T> m_promise {};
    std::future<T> m_future { m_promise.get_future() };
    std::coroutine_handle<> m_h {};
};

export template <>
class PromiseState<void> {
public:
    bool IsReady() const { return m_future.wait_for(std::chrono::milliseconds {}) == std::future_status::ready; }

    void SetValue()
    {
        m_promise.set_value();
        if (auto h = std::move(m_h)) {
            h();
        }
    }

    void GetValue() { m_future.get(); }

    void SetCoroutineHandle(std::coroutine_handle<> h)
    {
        if (IsReady()) {
            h();
        } else {
            m_h = std::move(h);
        }
    }

private:
    std::promise<void> m_promise {};
    std::future<void> m_future { m_promise.get_future() };
    std::coroutine_handle<> m_h {};
};

export template <typename T>
class Promise {
public:
    struct promise_type {
        Promise<T> get_return_object() { return m_pState; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void unhandled_exception() { }
        void return_value(T value) { m_pState->SetValue(std::move(value)); }

    private:
        std::shared_ptr<PromiseState<T>> m_pState { std::make_shared<PromiseState<T>>() };
    };

    static Promise<T> resolve(T t)
    {
        auto pState = std::make_shared<PromiseState<T>>();
        pState->SetValue(std::move(t));
        return std::move(pState);
    }

    Promise() = default;

    Promise(std::shared_ptr<PromiseState<T>> pState)
        : m_pState { std::move(pState) }
    {
    }

    bool await_ready() const { return m_pState->IsReady(); }

    void await_suspend(std::coroutine_handle<> h) { m_pState->SetCoroutineHandle(std::move(h)); }

    T await_resume() { return m_pState->GetValue(); }

    std::unique_ptr<std::shared_ptr<PromiseState<T>>> GetState()
    {
        return std::make_unique<std::shared_ptr<PromiseState<T>>>(m_pState);
    }

    static std::unique_ptr<std::shared_ptr<PromiseState<T>>> GetState(void* pUserData)
    {
        return std::unique_ptr<std::shared_ptr<PromiseState<T>>> { reinterpret_cast<std::shared_ptr<PromiseState<T>>*>(pUserData) };
    }

private:
    std::shared_ptr<PromiseState<T>> m_pState { std::make_shared<PromiseState<T>>() };
};

export template <>
class Promise<void> {
public:
    struct promise_type {
        Promise<void> get_return_object() { return m_pState; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void unhandled_exception() { }
        void return_void() { m_pState->SetValue(); }

    private:
        std::shared_ptr<PromiseState<void>> m_pState { std::make_shared<PromiseState<void>>() };
    };

    static Promise<void> resolve()
    {
        auto pState = std::make_shared<PromiseState<void>>();
        pState->SetValue();
        return std::move(pState);
    }

    Promise() = default;

    Promise(std::shared_ptr<PromiseState<void>> pState)
        : m_pState { std::move(pState) }
    {
    }

    bool await_ready() const { return m_pState->IsReady(); }

    void await_suspend(std::coroutine_handle<> h) { m_pState->SetCoroutineHandle(std::move(h)); }

    void await_resume() { m_pState->GetValue(); }

    std::unique_ptr<std::shared_ptr<PromiseState<void>>> GetState()
    {
        return std::make_unique<std::shared_ptr<PromiseState<void>>>(m_pState);
    }

    static std::unique_ptr<std::shared_ptr<PromiseState<void>>> GetState(void* pUserData)
    {
        return std::unique_ptr<std::shared_ptr<PromiseState<void>>> { reinterpret_cast<std::shared_ptr<PromiseState<void>>*>(pUserData) };
    }

private:
    std::shared_ptr<PromiseState<void>> m_pState { std::make_shared<PromiseState<void>>() };
};

}