import collections
import math
import operator
import os
import pprint
import shutil
import sys
import time

from http import HTTPStatus

from deepface import DeepFace
from PIL import Image

import numpy as np

import requests


QUEUE_NAME = os.environ.get("QUEUE_NAME", "identification")
WAIT_TIMEOUT = os.environ.get("WAIT_TIMEOUT", 30)
VIZAR_SERVER = os.environ.get("VIZAR_SERVER", "localhost:5000")
MIN_RETRY_INTERVAL = 5

DETECTOR_BACKEND = os.environ.get("DETECTOR_BACKEND", "mtcnn")
MODEL_NAME = os.environ.get("MODEL_NAME", "ArcFace")


def build_database():
    query_url = "http://{}/photos/annotations?label=person&identified_user_id".format(VIZAR_SERVER)
    response = requests.get(query_url)
    for note in response.json():
        user_dir = os.path.join("db", note['identified_user_id'])
        os.makedirs(user_dir, exist_ok=True)

        query_url = "http://{}/photos/{}/image".format(VIZAR_SERVER, note['photo_record_id'])
        response = requests.get(query_url)
        if not response.ok:
            continue

        content_type = response.headers.get("content-type")
        if content_type == "image/jpeg":
            extension = ".jpeg"
        elif content_type == "image/png":
            extension = ".png"
        else:
            continue

        output_filename = "image-{:08d}{}".format(note['photo_record_id'], extension)
        output_path = os.path.join(user_dir, output_filename)
        with open(output_path, "wb") as output:
            output.write(response.content)


def get_user_names():
    users = {}

    query_url = "http://{}/users".format(VIZAR_SERVER)
    response = requests.get(query_url)
    for user in response.json():
        users[user['id']] = user['display_name']

    return users


def run_recognition(img, users):
    detector_info = {
        "model_repo": "deepface",
        "model_name": "{}+{}".format(DETECTOR_BACKEND, MODEL_NAME),
        "engine_name": "deepface",
        "engine_version": "",
        "torchvision_version": "",
        "torch_version": "",
        "cuda_enabled": False,
        "preprocess_duration": 0,
        "inference_duration": 0,
        "nms_duration": 0,
        "postprocess_duration": 0
    }

    start_time = time.time()

    try:
        dfs = DeepFace.find(img, detector_backend=DETECTOR_BACKEND, model_name=MODEL_NAME, db_path="db", enforce_detection=False)
        if not isinstance(dfs, list):
            dfs = [dfs]
    except Exception as error:
        print(error)
        dfs = []

    end_time = time.time()
    detector_info['inference_duration'] = end_time - start_time

    image_height, image_width, _ = img.shape

    annotations = []
    for df in dfs:
        identity = df.at[0, "identity"]
        containing_dir = identity.split("/")[-2]

        confidence = 1.0 - df.at[0, "distance"]

        annotations.append({
            "identified_user_id": containing_dir,
            "label": "face",
            "sublabel": users.get(containing_dir, ""),
            "confidence": confidence,
            "boundary": {
                "left": float(df.at[0, "source_x"]) / image_width,
                "top": float(df.at[0, "source_y"]) / image_height,
                "width": float(df.at[0, "source_w"]) / image_width,
                "height": float(df.at[0, "source_h"]) / image_height,
            }
        })

    result = {
        "detector": detector_info,
        "annotations": annotations,
        "status": "done"
    }

    return result


def main():
    build_database()

    users = get_user_names()

    while True:
        sys.stdout.flush()

        query_url = "http://{}/photos?queue_name={}&wait={}".format(VIZAR_SERVER, QUEUE_NAME, WAIT_TIMEOUT)
        start_time = time.time()

        items = []

        try:
            response = requests.get(query_url)
            if response.ok and response.status_code == HTTPStatus.OK:
                items = response.json()
        except requests.exceptions.RequestException as error:
            # Most common case is if the API server is restarting,
            # then we see a connection error temporarily.
            print(error)

        # Check if the empty/error response from the server was sooner than
        # expected.  If so, add an extra delay to avoid spamming the server.
        # We need this in case long-polling is not working as expected.
        if len(items) == 0:
            elapsed = time.time() - start_time
            if elapsed < MIN_RETRY_INTERVAL:
                time.sleep(MIN_RETRY_INTERVAL - elapsed)
            continue

        for item in items:
            # Sort by priority level (descending), then creation time (ascending)
            item['priority_tuple'] = (-1 * item.get("priority", 0), item.get("created"))

        items.sort(key=operator.itemgetter("priority_tuple"))
        for item in items:
            url = "http://{}/photos/{}/image".format(VIZAR_SERVER, item['id'])
            print(url)

            req = requests.get(url, stream=True, timeout=60)
            img = np.array(Image.open(req.raw))

            # Drop the alpha channel
            if img.shape[2] == 4:
                img = img[:, :, 0:3]

            # Convert RGB to BGR
            img = img[:, :, ::-1]

            result = run_recognition(img, users)

            url = "http://{}/photos/{}".format(VIZAR_SERVER, item['id'])
            print(url)

            pprint.pprint(result)
            req = requests.patch(url, json=result)


if __name__ == "__main__":
    main()
