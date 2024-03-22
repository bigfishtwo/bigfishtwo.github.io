---
title: Python 装饰器原理和用法
date: 2024-03-21 +0800
categories: [代码笔记]
tags: [Python]
math: true
---

装饰器可以在不修改原有代码的情况下，为函数增加新功能或者限制条件。装饰器是基于闭包环境存在的。

闭包（Closure）指的是一个函数及其执行环境（变量）的组合。闭包需要满足：
- 内嵌函数
- 内嵌函数中有外部嵌套函数的变量
- 外部嵌套函数的返回值是内嵌函数名

装饰器函数可以使用装饰器语法糖 @，也可以采用原始方式进行调用。

被装饰函数的参数取决于装饰器函数的参数。
装饰器函数应该接收一个函数作为参数，并返回一个新的函数，新函数会替换原来的函数，从而实现对原有代码功能的动态修改。

一些装饰器用法的展示：

### 1. 日志打印

```python
################ 闭包环境 ##############
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"准备开始执行：{func.__name__} 函数:")
        result = func(*args, **kwargs)
        print("执行结束。")
        return result
    return wrapper
#######################################

@logger
def add(x, y):
    print(f"{x} + {y} = {x + y}")

add(200, 50)  

# 输出：
# 准备开始执行：add 函数:
# 200 + 50 = 250
# 执行结束。
```

### 2. 函数计时

```python
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        cost_time = end_time - start_time
        print(f"花费时间：{cost_time:.4f}秒")
        return result
    return wrapper

@timer
def sleep(seconds):
    time.sleep(seconds)

sleep(10)  

# 输出：
#  花费时间：10.0127秒
```

### 3. 带参数的函数装饰器

```python
def say_hello(country):  # 嵌套了两层
    def decorator(func):
        def wrapper(*args, **kwargs):
            if country == "china":
                print("你好！")
            elif country == "america":
                print("hello.")
            else:
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator

@say_hello("china")
def xiaoming():
    pass

@say_hello("america")
def jack():
    pass

xiaoming()  # 输出：你好！
jack()  # 输出：hello.
```

### 4. 不带参数的类装饰器

```python
class Logger:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print(f"[INFO]: 函数 {self.func.__name__}() 正在运行...")
        return self.func(*args, **kwargs)

@Logger
def say(something):
    print(f"say {something}!")

say("hello")  


# 输出：
# [INFO]: 函数 say() 正在运行... 
#  say hello!
```

### 5. 多个装饰器对同一个函数进行装饰

```python
def set_function1(fun):
    print('---开始进行装饰1---')
    def call_function1(*args, **kwargs):
        print('---权限1验证---')
        print(args, kwargs)
        result = 100 + fun(*args, **kwargs)
        print(f"装饰1 返回值：{result}")
        return result
    return call_function1
def set_function2(fun):
    print('---开始进行装饰2---')
    def call_function2(*args, **kwargs):
        print('---权限2验证---')
        print(args, kwargs)
        result = 200 + fun(*args, **kwargs)
        print(f"装饰2 返回值：{result}")
        return result
    return call_function2
@set_function1
@set_function2
def test1(num, *args, **kwargs):
    print('---test1---')
    print(num, args, kwargs)
    return num
ret = test1(100)
print(ret)

# 输出结果：
# ---开始进行装饰2---
# ---开始进行装饰1---
# ---权限1验证---
# (100,) {}
# ---权限2验证---
# (100,) {}
# ---test1---
# 100 () {}
# 装饰2 返回值：300
# 装饰1 返回值：400
# 400
```

这里可以看出在执行到 `@set_function1`的时候就开始装饰了，而不是`test1()`运行的时候。
`test1` 函数会被 `set_function2` 装饰器装饰，然后再被 `set_function1` 装饰器装饰。所以先输出“开始进行装饰2”，再输出“开始进行装饰1”。

`test1` 函数的执行顺序是：`set_function1` -> `set_function2` -> `test1`。
计算过程是 `call_function1` 接受到被 2 装饰的 `test1`，于是执行  `call_function2`，因此两个函数打印出来的参数相同。最后是调用被装饰的`test1`，打印输入值 100，返回输出值 100。装饰 2 执行了 +200 的操作，返回 300。接着装饰 1 执行 +100 的操作，返回 400。

最终返回值 400。


### 参考：

- https://zhuanlan.zhihu.com/p/76056230