import json
from http import HTTPStatus

from flask import Blueprint, request

from environment.state_handling import get_storage_path, is_multi_fp_collection
from utilities.metrics import write_resource_metrics_to_file, write_syscall_metrics_to_file

fp_bp = Blueprint("fingerprint", __name__, url_prefix="/fp")


@fp_bp.route("/<mac>", methods=["POST"])
def report_fingerprint(mac):

    resource_fp = json.loads(request.form.get('resource_fp'))

    # resource fingerprint
    write_resource_metrics_to_file(str(resource_fp["rate"]), str(resource_fp["fp"]), get_storage_path(),
                                   is_multi_fp_collection())

    # syscall data
    if "syscalls" not in request.files:
        return "syscall file was not sent correctly", HTTPStatus.BAD_REQUEST
    raw_syscall_file = request.files["syscalls"]
    if raw_syscall_file.filename == "":
        return "syscall file was not sent correctly", HTTPStatus.BAD_REQUEST

    write_syscall_metrics_to_file(raw_syscall_file)

    return "", HTTPStatus.CREATED
