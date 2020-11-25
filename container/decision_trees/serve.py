import os
import json
import pickle
import sys
import subprocess
import traceback
from collections import OrderedDict
from typing import List
import yaml
import numpy as np
import pandas as pd
from pydantic import BaseModel, root_validator
from fastapi import FastAPI, HTTPException


class Config:
    def __init__(self, file_path):
        try:
            self.config_file = file_path
            with open(config_file_path) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.project_name = self.config["project_name"]
            self.prefix = self.config[
                "dir_root"
            ]  # TODO: Change this to /opt/ml
            self.model_dir = os.path.join(
                self.prefix, self.config["model_dir_prefix"]
            )
            self.default_param_path = os.path.join(
                "", self.config["default_hyperparam_file_name"]
            )
            # self.model_subdir = os.path.join(
            #     self.model_dir, self.config["project_name"]
            # )
            with open(
                os.path.join(self.model_dir, "training_metadata.yaml")
            ) as f:
                self.latest_training_metadata = yaml.load(
                    f, Loader=yaml.FullLoader
                )
            self.arch = self.latest_training_metadata["arch"]
            self.model_path = self.latest_training_metadata["model_path"]
            self.num_classes = self.latest_training_metadata["num_classes"]            
        except Exception as e:
            msg = (
                f"Exception raised while loading configuration:\n{e}\n"
                f"{traceback.format_exc()}"
            )
            raise Exception(msg)

    def health_check(self):
        try:
            if self.latest_training_metadata["training_status"] != "Success":
                msg = "Last training did not finish successfully. Please reinitiate training"
                raise Exception(msg)
                return {"status": False, "msg": msg}
            for i in [
                self.model_dir,
                # self.model_subdir,
            ]:
                if not os.path.isdir(i):
                    msg = f"Directory could not be found: {i}"
                    raise Exception(msg)
                    return {"status": False, "msg": msg}

            if not os.path.isfile(self.default_param_path):
                msg = (
                    f"Default hyperparameters file does not exist at: {self.default_param_path}. "
                    f"Please supply a valid {self.default_param_path} in the docker container."
                )
                raise Exception(msg)
                return {"status": False, "msg": msg}
            if not os.path.isfile(self.model_path):
                msg = (
                    f"Model file absent at: {self.model_path}. "
                    f"Please reinitiate training."
                )
                raise Exception(msg)
                return {"status": False, "msg": msg}

            if not self.arch:
                msg = "missing latest_arch; reinitiate training"
                raise Exception(msg)
                return {"status": False, "msg": msg}            
            try:
                if self.arch == 'decision_tree':
                    with open(self.model_path, 'rb') as inp:
                        self.model =  pickle.load(inp)
                # self.model = self.model.cpu()
                # self.model.load_state_dict(
                #     torch.load(self.model_path, map_location=torch.device('cpu'))
                # )
            except Exception as e:
                msg = f"exception raised during loading  model: {e}"
                raise Exception(msg)
                return {"status": False, "msg": msg}
            return {"status": True}
        except Exception as e:
            raise e

class Iris(BaseModel):
    sepals_length: List[float]
    sepals_width: List[float]
    petals_length: List[float]
    petals_width: List[float]
    @root_validator
    def check_dimensions(cls, values):
        assert len(values.get("sepals_length")) == len(
            values.get("sepals_width")
        ), "sepals length-width mismatch"
        assert len(values.get("sepals_length")) == len(
            values.get("petals_length")
        ), "sepals length petals length mismatch"
        assert len(values.get("petals_length")) == len(
            values.get("petals_width")
        ), "petals length width mismatch"
        return values

app = FastAPI()

config_file_path = os.path.join("", "config.yaml")
try:
    config = Config(config_file_path)
    health = config.health_check()
except Exception as e:
    print(f'Exception raised during health-check:\n{e}', file=sys.stderr)
    sys.exit(1)

@app.get("/ping")
async def check_health():
    """
    This is the first of the two end points required by AWS Sagemaker. 
    This checks the overal condition of the project directories within 
    the container, reports the necessary paths for predictions and raises 
    exception in case of invalid configurations. The health checks ends 
    in successfully loading the pytorch model from the latest training 
    phase
    """
    print(health)
    try:
        if not health["status"]:
            print(
                f"Exception during health-check:\n{health['msg']}",
                file=sys.stderr,
            )
            sys.exit(1)
            raise HTTPException(status_code=404, detail=health["msg"])
    
        else:
            return {"msg": ""} # Return an empty payload to pass the ping test
            #         return health["msg"]
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


@app.post("/invocations")
async def predict(iris: Iris):
    try:
        print("Hello World")
        sepals_length = np.array(iris.sepals_length, dtype=np.float)
        sepals_width = np.array(iris.sepals_width, dtype=np.float)
        petals_length = np.array(iris.petals_length, dtype=np.float)
        petals_width = np.array(iris.petals_width, dtype=np.float)
        data = np.column_stack((sepals_length, sepals_width,
                        petals_length, petals_width))
        prediction = config.model.predict(data)        
        
    except Exception as e:
        print(f"Exception raised during invocations:{e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"{e}")

    else:
        print("Success")
        return {            
            "iris_class": prediction.flatten().tolist()            
        }        
