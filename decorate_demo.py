from time import time


def print_time(arg):
    def f_dec(fun):
        def t_dec(x, y):
            print('**'*10, time())
            print(arg)
            return fun(x, y)
        return t_dec
    return f_dec


# add = print_time('there you go')(add)
# add = f_dec(add)
# add = t_dec
# add(x, y) = t_dec(x, y)
@print_time('there you go')
def add(x, y):
    print(x+y)

# add(1, 2)


def mention(fun):
    def wrapper(*args, **kwargs):
        print('ICBC welcome you.')
        print('ICBC:', fun(*args, **kwargs))
    return wrapper


@mention
def add(*args, **kwargs):
    print('args:', args, '\n*args', *args)
    return sum(args)


add(100, 20, 30, 50, 100)


