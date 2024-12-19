import os
import torch
from flask import Flask
from flask_cors import CORS
from .config.deploy_config import config as dep_config
from .config.system_config import  PoseEstimatorConfig

from .components.ext_lib import models
from .components.ext_lib.core.config import config
from .components.ext_lib.core.config import update_dir
from .components.ext_lib.core.config import update_config
from .components.ext_lib.core.config import get_model_name


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_app(env_config=None):
    # instantiate the app
    app = Flask(__name__)
    cors = CORS(app)

    ###### ENVIROMENT VARIABLE CONFIGURATION #######################
    if env_config is None:
        env_config = os.getenv("PROD_APP_SETTINGS", "development")
    app.config.from_object(dep_config[env_config])

    
    ###### MODEL INITIALIZATION #####################################
    update_config(PoseEstimatorConfig.config_path)
    config.GPUS = ''
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(config, is_train=False)
    model.load_state_dict(torch.load(PoseEstimatorConfig.pe_model_path, map_location=torch.device('cpu')))

    PoseEstimatorConfig.pe_model = model
    ##### REGISTER REFLECTLY_LLM BLUEPRINT ENDPOINTS ################
    from .pe_connect import pe_endpoints
    app.register_blueprint(pe_endpoints)

    # shell context for flask cli
    @app.shell_context_processor
    def ctx():
        return {"app": app}
    return app