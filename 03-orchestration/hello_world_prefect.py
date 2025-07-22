# hello_prefect.py

from prefect import flow, task

@task
def say_hello(name: str = "world"):
    print(f"Hello, {name}!")

@flow
def hello_flow():
    say_hello()

if __name__ == "__main__":
    hello_flow()
