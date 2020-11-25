import sys
import subprocess
import typer
from gunicorn_conf import use_bind, web_concurrency

app = typer.Typer()
python_executable = sys.executable
training_script = "train.py"
server_script = "serve.py"
# server_command = [
#     "gunicorn",
#     "serve:app",
#     "-b",
#     f"{use_bind}",
#     "-w",
#     f"{web_concurrency}",
#     "-k",
#     "uvicorn.workers.UvicornH11Worker",
# ]
server_command = [
    "uvicorn",
    "serve:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8080",
]

# server_command = [
#     "uvicorn",
#     "serve:app",
#     "--port",
#     "8889",
# ]

@app.command()
def train():
    training_process = subprocess.Popen(
        [python_executable, training_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_value, stderr_value = training_process.communicate()
    print(stdout_value.decode("utf-8"))
    print(stderr_value.decode("utf-8"))
    return_code = training_process.poll()
    if return_code:
        error_msg = (
            f"Exception raised during model training:\n"
            f"Return code: {return_code}:\n"
            f'stderr output: {stderr_value.decode("utf-8")}'
        )
        raise Exception(error_msg)


@app.command()
def serve():
    server_process = subprocess.Popen(
        server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout_value, stderr_value = server_process.communicate()
    print(stdout_value.decode("utf-8"))
    print(stderr_value.decode("utf-8"))
    return_code = server_process.poll()
    if return_code:
        error_msg = (
            f"Exception raised during model serving:\n"
            f"Return code: {return_code}:\n"
            f'stderr output: {stderr_value.decode("utf-8")}'
        )
        raise Exception(error_msg)


if __name__ == "__main__":
    app()
