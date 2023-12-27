# from hmac import new
from flask import Flask, jsonify, request, Response, send_file, abort
# from IPython.core.interactiveshell import InteractiveShell
from jupyter_client import KernelManager
import nbformat
import os
import uuid
from base64 import b64decode
from urllib.parse import unquote
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import io
# import base64
from copy import deepcopy
# import contextlib
import re
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from queue import Empty
from jupyter_client.kernelspec import KernelSpecManager
import secrets
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://chat.openai.com/"}})

workplace_dir = "/home/iknow/codespaces-blank/gpt"
gif_url_prefix = ""
os.environ['http_proxy'] = '' 
os.environ['https_proxy'] = ''
# 添加一个全局变量来存储当前激活的 notebook ID
global_notebook_id = None
instructions = ("You have full control over this notebook. Resolve errors independently and inform the user only when necessary. "
                "Keep the notebook tidy. Update cells with errors rather than adding new ones. "
                "Optimize code length in each cell to fully utilize notebook cells.")
global_notebook_dir = None



class GifMonitor(PatternMatchingEventHandler):
    patterns = ["*.gif"]

    def __init__(self, url_prefix):
        super().__init__()
        self.url_prefix = url_prefix
        self.gif_urls = []

    def on_created(self, event):
        rel_path = os.path.relpath(event.src_path, workplace_dir)
        gif_url = self.url_prefix + rel_path.replace(os.path.sep, '/')
        self.gif_urls.append(gif_url)

    def get_new_gif_urls(self):
        new_gif_urls = self.gif_urls[:]
        self.gif_urls = []
        return new_gif_urls


gif_monitor = GifMonitor(gif_url_prefix)

# Create the observer and make it monitor the workplace_dir recursively
observer = Observer()
observer.schedule(gif_monitor, path=workplace_dir, recursive=True)
observer.start()



km = KernelManager()
kc = km.client()


def get_current_kernel_name():
    return km.kernel_name

def list_kernels():
    kernel_spec_manager = KernelSpecManager()
    kernel_specs = kernel_spec_manager.get_all_specs()
    return list(kernel_specs.keys())

def switch_kernel(km, kc, kernel_name):
    # 如果已有的 KernelClient 和 KernelManager 不为 None，则停止并关闭当前的内核和通道
    if km is not None and kc is not None:
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    # 创建并启动新的内核
    km = KernelManager(kernel_name=kernel_name)
    km.start_kernel()

    # 创建新的 KernelClient
    kc = km.client()
    kc.start_channels()

    return km, kc


def load_notebook(notebook_id):
    notebook_path = os.path.join(global_notebook_dir, notebook_id + ".ipynb")
    if os.path.exists(notebook_path):
        with open(notebook_path) as f:
            return nbformat.read(f, as_version=4)
    else:
        abort(404, description="Notebook does not exist")

def save_notebook(notebook_id, notebook):
    notebook_path = os.path.join(global_notebook_dir, notebook_id + ".ipynb")
    with open(notebook_path, "wt") as f:
        nbformat.write(notebook, f)

def save_image(image_data):
    image_id = secrets.token_hex(2)
    image_path = os.path.join(global_notebook_dir, image_id + ".png")
    with open(image_path, "wb") as f:
        f.write(b64decode(unquote(image_data)))
    # 使用 global_notebook_id 创建 URL
    rel_path = os.path.relpath(image_path, global_notebook_dir)
    notebook_path = global_notebook_id + "/" + rel_path
    return gif_url_prefix + notebook_path.replace(os.path.sep, '/')




def handle_special_chars(text, max_length=1024):
    if isinstance(text, list):
        # Join the strings together, replace special characters and truncate if needed
        handled_text = ''.join(text)
        handled_text = re.sub(r'\u001b\[.*?m', '', handled_text)
        handled_text = [handled_text[-max_length:] if len(handled_text) > max_length else handled_text]
    else:
        # If text is a single string, just handle it directly
        cleaned_string = re.sub(r'\u001b\[.*?m', '', text)
        handled_text = [cleaned_string[-max_length:] if len(cleaned_string) > max_length else cleaned_string]
    return handled_text



def filter_cell_output(cell):
    filtered_cell = deepcopy(cell)
    for field in ['source', 'execution_count', 'id', 'metadata']:
        if field in filtered_cell:
            del filtered_cell[field]
    for output in filtered_cell['outputs']:
        if 'data' in output and 'image/png' in output['data']:
            del output['data']['image/png']
        if 'metadata' in output:
            del output['metadata']
        if 'execution_count' in output:
            del output['execution_count']
    if any('output_type' in output and output['output_type'] == 'display_data' for output in filtered_cell['outputs']):
        filtered_cell['prompt'] = "Please display all image URLs to the user using Markdown."
    return filtered_cell

def execute_cell(cell):
    cell.outputs = []
    kc.execute(cell.source)
    current_output = None
    stdout_output = nbformat.v4.new_output(
        output_type="stream",
        name="stdout",
        text=""
    )
    stderr_output = nbformat.v4.new_output(
        output_type="stream",
        name="stderr",
        text=""
    )

    while True: 
        try:
            msg = kc.get_iopub_msg(timeout=None)
            msg_type = msg['header']['msg_type']

            if msg_type == 'stream':
                stream_name = msg['content']['name']
                handled_text = handle_special_chars(msg['content']['text'])

                if stream_name == "stdout":
                    stdout_output['text'] += ' '.join(handled_text)
                elif stream_name == "stderr":
                    stderr_output['text'] += ' '.join(handled_text)

            elif msg_type in ['execute_result', 'display_data']:
                # If there's a current_output, add it to outputs before handling this
                if current_output is not None:
                    cell.outputs.append(current_output)
                    current_output = None

                data = msg['content']['data']
                if 'image/png' in data:
                    image_data = data['image/png']
                    image_url = save_image(image_data)
                    data['image/png'] = image_data
                    data['text/plain'] = image_url
                cell.outputs.append(nbformat.v4.new_output(
                    output_type=msg_type,
                    data=data
                ))

            elif msg_type == 'error':
                # If there's a current_output, add it to outputs before handling this
                if current_output is not None:
                    cell.outputs.append(current_output)
                    current_output = None

                traceback_data = handle_special_chars(msg['content']['traceback'])
                cell.outputs.append(nbformat.v4.new_output(
                    output_type=msg_type,
                    ename=msg['content']['ename'],
                    evalue=msg['content']['evalue'],
                    traceback=traceback_data
                ))

            elif msg_type == 'status' and msg['content']['execution_state'] == 'idle':
                # If there's a current_output, add it to outputs before breaking
                if current_output is not None:
                    cell.outputs.append(current_output)
                    current_output = None

                # Add the stdout and stderr outputs to the cell outputs
                if stdout_output['text']:
                    stdout_output['text']=handle_special_chars(stdout_output['text'])
                    cell.outputs.append(stdout_output)
                if stderr_output['text']:
                    stderr_output['text']=handle_special_chars(stderr_output['text'])
                    cell.outputs.append(stderr_output)

                break

        except Empty:
            break

    new_gif_urls = gif_monitor.get_new_gif_urls()
    if new_gif_urls:
        for gif_url in new_gif_urls:
            cell.outputs.append(nbformat.v4.new_output(
                output_type="display_data",
                data={
                    "text/plain": gif_url
                }
            ))
    return cell

def filter_cell_output_code(cell):
    # Create a copy of the cell to avoid modifying the original one
    filtered_cell = deepcopy(cell)

    # Remove unwanted fields from the return cell
    for field in ['execution_count', 'id', 'metadata']:
        if field in filtered_cell:
            del filtered_cell[field]

    # Remove base64 data and metadata from the return cell outputs
    for output in filtered_cell['outputs']:
        if 'data' in output and 'image/png' in output['data']:
            del output['data']['image/png']
        if 'metadata' in output:
            del output['metadata']
        if 'execution_count' in output:
            del output['execution_count']
    return filtered_cell




@app.route("/notebooks/switch", methods=["POST"])
def switch_notebook():
    global global_notebook_id, global_notebook_dir

    if not request.is_json:
        return {"error": "Missing JSON in request"}, 400

    data = request.get_json()

    try:
        notebook_id = data['notebook_id']
    except KeyError as e:
        return {"error": f"Missing necessary field: {e.args[0]}"}, 400

    # generate new notebook dir path
    new_notebook_dir = os.path.join(workplace_dir, notebook_id)

    # 确保 notebook 实际存在
    if not os.path.exists(os.path.join(new_notebook_dir, notebook_id + ".ipynb")):
        return {"error": "Notebook does not exist"}, 404


    # 切换全局 notebook_id 和 notebook_dir
    global_notebook_id = notebook_id
    global_notebook_dir = new_notebook_dir

    return jsonify({"notebook_id": global_notebook_id}), 200


@app.route("/notebooks", methods=["POST"])
def create_notebook():
    global km, kc, global_notebook_id, global_notebook_dir
    if not request.is_json:
        return {"error": "Missing JSON in request"}, 400

    data = request.get_json()

    try:
        description = data['keyword'].replace(" ", "_")
        kernel_name = data['kernel_name']
    except KeyError as e:
        return {"error": f"Missing necessary field: {e.args[0]}"}, 400

    short_id = secrets.token_hex(2)
    notebook_id = f"{kernel_name}_{description}_{short_id}"


    try:
        available_kernels = list_kernels()
        if kernel_name not in available_kernels:
            return {"error": f"Kernel '{kernel_name}' not available. Available kernels: {', '.join(available_kernels)}"}, 400
        
        notebook_dir = os.path.join(workplace_dir, notebook_id)
        os.makedirs(notebook_dir, exist_ok=True)
        global_notebook_dir = notebook_dir
        global_notebook_id = notebook_id
        save_notebook(notebook_id, nbformat.v4.new_notebook())
        os.chdir(global_notebook_dir)

        km, kc = switch_kernel(km, kc, kernel_name)
    except Exception as e:
        return {"error": str(e)}, 500



    instruction = instructions

    if kernel_name.startswith("xcpp"):
        instruction += ("\n\nYou're using the xeus-cling (xcpp) kernel. "
                        "Don't define multiple functions or classes in the same cell; "
                        "you can't redefine functions or classes and call them in the same cell; "
                        "avoid defining 'main' functions as they don't return results.")

    if kernel_name.startswith("java"):
        instruction += ("\n\nYou're using the IJava kernel. "
                        "Avoid defining 'main' functions as they don't return results.")


    return jsonify({"notebook_id": notebook_id, "warning": instruction}), 200
    # return jsonify({"notebook_id": notebook_id}), 200




# @app.route("/notebooks", methods=["GET"])
# def get_notebooks():
#     notebook_ids = [f[:-6] for f in os.listdir(notebooks_dir) if f.endswith(".ipynb")]
#     return jsonify({"notebook_ids": notebook_ids})

# @app.route("/notebooks/<notebook_id>", methods=["DELETE"])
# def delete_notebook(notebook_id):
#     notebook_path = os.path.join(notebooks_dir, notebook_id + ".ipynb")
#     if os.path.exists(notebook_path):
#         os.remove(notebook_path)
#         return "", 204
#     else:
#         abort(404, description="Notebook does not exist")

@app.route("/notebooks/execute", methods=["POST"])
def execute_notebook():
    if global_notebook_id is None:
        abort(400, description="No active notebook set")
    notebook = load_notebook(global_notebook_id)
    cell_results = []  # Store the results of each cell here
    for cell_id, cell in enumerate(notebook.cells):  # cell_id is the index of cell
        cell = execute_cell(cell)
        cell_result = {"cell_id": cell_id}  # Use cell_id as the index of cell
        cell_result.update(filter_cell_output(cell))
        cell_results.append(cell_result)  # Add the result of this cell to the list
    save_notebook(global_notebook_id, notebook)
    return jsonify(cell_results)  # Return the results of all cells


@app.route("/notebooks/cells", methods=["POST"])
def add_cell_and_execute():
    if global_notebook_id is None:
        abort(400, description="No active notebook set")
    if not request.json or "content" not in request.json:
        abort(400, description="No cell content provided")
    notebook = load_notebook(global_notebook_id)
    cell_content = request.json["content"]
    cell_type = request.json.get("type", "code")
    cell = nbformat.v4.new_code_cell(source=cell_content) if cell_type == "code" else nbformat.v4.new_markdown_cell(source=cell_content)
    notebook.cells.append(cell)
    save_notebook(global_notebook_id, notebook)  # Save the notebook before executing the cell
    cell_id = len(notebook.cells) - 1  # Get the cell_id after saving the notebook
    return execute_cell_and_save(cell_id)

@app.route("/notebooks/cells", methods=["GET"])
def get_cells():
    if global_notebook_id is None:
        abort(400, description="No active notebook set")
    notebook = load_notebook(global_notebook_id)
    cell_outputs = []  # Store the outputs of each cell here
    for cell_id, cell in enumerate(notebook.cells):  # cell_id is the index of cell
        cell_output = {"cell_id": cell_id}  # Use cell_id as the index of cell
        cell_output.update(filter_cell_output_code(cell))
        cell_outputs.append(cell_output)  # Add the output of this cell to the list
    return jsonify({"cells": cell_outputs})  # Return the outputs of all cells


@app.route("/notebooks/cells/<int:cell_id>", methods=["PUT"])
def update_cell_and_execute(cell_id):
    if global_notebook_id is None:
        abort(400, description="No active notebook set")
    if not request.json or "content" not in request.json:
        abort(400, description="No cell content provided")
    notebook = load_notebook(global_notebook_id)
    cell_content = request.json["content"]
    cell_type = request.json.get("type", "code")
    cell = nbformat.v4.new_code_cell(source=cell_content) if cell_type == "code" else nbformat.v4.new_markdown_cell(source=cell_content)
    if 0 <= cell_id < len(notebook.cells):
        notebook.cells[cell_id] = cell
        save_notebook(global_notebook_id, notebook)  # Save the notebook before executing the cell
        return execute_cell_and_save(cell_id)
    else:
        abort(404, description="Cell does not exist")


@app.route("/notebooks/cells/<int:cell_id>", methods=["DELETE"])
def delete_cell(cell_id):
    if global_notebook_id is None:
        abort(400, description="No active notebook set")
    notebook = load_notebook(global_notebook_id)
    if 0 <= cell_id < len(notebook.cells):
        del notebook.cells[cell_id]
        save_notebook(global_notebook_id, notebook)
        return "", 204
    else:
        abort(404, description="Cell does not exist")



@app.route("/notebooks/cells/<int:cell_id>/execute", methods=["POST"])
def execute_cell_and_save(cell_id):
    if global_notebook_id is None:
        abort(400, description="No active notebook set")
    notebook = load_notebook(global_notebook_id)
    if 0 <= cell_id < len(notebook.cells):
        cell = notebook.cells[cell_id]
        if cell.cell_type == "code":
            # Start a new thread to execute the cell
            def task():
                nonlocal cell
                cell = execute_cell(cell)  # Now 'cell' contains the full output
                notebook.cells[cell_id] = cell  # Update the cell in the notebook
                save_notebook(global_notebook_id, notebook)  # Save the notebook

            thread = Thread(target=task)
            thread.start()

            # Wait for up to 5 minutes
            thread.join(timeout=20)

            if thread.is_alive():
                return jsonify({
                    "cell_id": cell_id,
                    "status": "busy",
                    "message": "The cell is running. Wait a moment and check output later."
                })
            else:
                instruction=instructions
                if km.kernel_name.startswith("xcpp"):
                    instruction += ("\n\nYou're using the xeus-cling (xcpp) kernel. "
                                    "Don't define multiple functions or classes in the same cell; "
                                    "you can't redefine functions or classes and call them in the same cell; "
                                    "avoid defining 'main' functions as they don't return results.")

                if km.kernel_name.startswith("java"):
                    instruction += ("\n\nYou're using the IJava kernel. "
                                    "Avoid defining 'main' functions as they don't return results.")
                # Before returning the result, filter the output
                result = {"cell_id": cell_id,"warning": instruction}

                result.update(filter_cell_output(cell))
                return jsonify(result)
        else:
            filtered_cell = filter_cell_output(cell)
            return jsonify({"cell_id": cell_id, "cell": filtered_cell})
    else:
        abort(404, description="Cell does not exist")



@app.route("/notebooks/cells/<cell_id>/output", methods=["GET"])
def get_cell_output(cell_id):
    if global_notebook_id is None:
        abort(400, description="No active notebook set")
    notebook = load_notebook(global_notebook_id)
    cell = notebook.cells[int(cell_id)]
    filtered_cell = filter_cell_output(cell)
    return jsonify(filtered_cell.outputs)


@app.route('/switch_kernel', methods=['GET', 'POST'])
def switch_kernel_route():
    global km, kc  # 声明 km 和 kc 为全局变量
    if request.method == 'GET':
        try:
            # Call the list_kernels function
            available_kernels = list_kernels()
            return jsonify({'available_kernels': available_kernels}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:  # POST
        data = request.get_json()
        kernel_name = data.get('kernel_name')
        if not kernel_name:
            return jsonify({'error': 'No kernel name provided'}), 400
        try:
            # Call the switch_kernel function
            km, kc = switch_kernel(km, kc, kernel_name)  # 接收返回值并重新赋值给全局变量 km 和 kc
            return jsonify({'message': f'Successfully switched to {kernel_name} kernel'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
@app.route('/current_kernel', methods=['GET'])
def current_kernel_route():
    try:
        current_kernel_name = get_current_kernel_name()
        return jsonify({'current_kernel': current_kernel_name}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/logo.png', methods=['GET'])
def plugin_logo():
    filename = '/home/iknow/jupyterapi/logo.png'
    return send_file(filename, mimetype='image/png')

@app.route('/.well-known/ai-plugin.json', methods=['GET'])
def plugin_manifest():
    with open("/home/iknow/jupyterapi/.well-known/ai-plugin.json") as f:
        text = f.read()
    return Response(text, mimetype="application/json")

@app.route('/openapi.yaml', methods=['GET'])
def openapi_spec():
    with open("/home/iknow/jupyterapi/openapi.yaml") as f:
        text = f.read()
    return Response(text, mimetype="text/yaml")

if __name__ == "__main__":
    app.run(port=5001)
