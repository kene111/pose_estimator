from flask import Blueprint

pe_endpoints = Blueprint('pe_endpoints', __name__)
import os
import json

from flask_cors import cross_origin
from werkzeug.utils import secure_filename
from .components.pose_estimator import PoseEstimator
from .config.system_config import PoseEstimatorConfig
from .utils.system_utils import construct_downloadble_uri
from flask import jsonify, request, Response, send_from_directory,  send_file


@pe_endpoints.route('/pe_alive', methods=['GET'])
@cross_origin()
def pe_alive():
    pe_response = {"system_message":"I am alive!"}
    return Response(response=json.dumps(pe_response), status=200, mimetype='application/json')

@pe_endpoints.route('/estimate_pose', methods=['POST'])
@cross_origin()
def pose_estimation():
    pe_estimator = PoseEstimator()
    file_ = request.files['file']

    filename = secure_filename(file_.filename)
    image_path =  os.path.join(PoseEstimatorConfig.uploaded_dir, filename)
    file_.save(image_path)

    try:
        _, image_path = pe_estimator.estimate_pose(image_path, save=True)
        image_download_uri = construct_downloadble_uri(image_path)
        system_response = {"system_message": image_download_uri}
        return Response(response=json.dumps(system_response), status=200, mimetype='application/json')
    except Exeption as e:
        system_response = {"system_message":f"System Error: {e}"}
        return Response(response=json.dumps(system_response), status=400, mimetype='application/json')


@pe_endpoints.route("/download/<path:file_name>", methods=['GET'])
@cross_origin()
def download_file(file_name):
    return send_from_directory(directory=PoseEstimatorConfig.output_dir, path=file_name, as_attachment=True)