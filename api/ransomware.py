from http import HTTPStatus

from flask import Blueprint

rw_bp = Blueprint("ransomware", __name__, url_prefix="/rw")


@rw_bp.route("/done", methods=["PUT"])
def mark_done():
    return "", HTTPStatus.NO_CONTENT