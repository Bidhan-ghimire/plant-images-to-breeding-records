"""Supporting functions for the strawberry runner demonstration notebook."""
# Supporting implementation for the short calls in runner_to_breedbase.ipynb.
from pathlib import Path
import os, sys, json
from html import escape
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, HTML, SVG

RUN_PREDICTIONS = False
DO_UPLOAD = False
DO_RETRIEVE = False
PREDICTION_RUN = None  # Optional path to a run generated on this computer.
BRAPI_BASE = "http://localhost:8099/brapi/v2"

candidates = [Path(__file__).resolve().parent, Path.cwd(), *Path.cwd().parents]
candidates += [path / "breedbase_extension" for path in list(candidates)]
EXTENSION = next((path for path in candidates
                  if (path / "bb_records/plant_demo/plant_image_mapping.csv").is_file()
                  ), None)
if EXTENSION is None:
    raise RuntimeError("Open this notebook from the strawberry-runner project folder.")
sys.dont_write_bytecode = True
demo = None
rows = []
frame = None
payloads = []
upload_result = None
retrieval_result = None
client = None

# Image verification and optional model inference.
"""Verified local plant/image data; optional inference is explicit and never runs on import."""
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import csv
import hashlib
import json
import math

CHECKPOINT_SHA256 = "69b679d12f76ccfc99e692e8481815eb2ca2c2e798126257152f76f9908152fc"
SETTINGS = dict(imgsz=320, conf=0.25, iou=0.7, max_det=300,
                augment=False, half=False, device="cuda:0")
VERSIONS = {"torch": "2.10.0+cu128", "torchvision": "0.25.0+cu128",
            "ultralytics": "8.4.9", "pandas": "2.3.3", "Pillow": "12.0.0",
            "ipykernel": "7.3.0", "jupyterlab": "4.6.4"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _integer(value, field):
    try:
        number = Decimal(str(value))
        if isinstance(value, bool) or not number.is_finite() or number < 0 or number != number.to_integral_value():
            raise ValueError()
        return int(number)
    except (ValueError, InvalidOperation):
        raise ValueError(f"{field} must be a present, nonnegative integer; got {value!r}") from None


def _boolean(value, field):
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"{field} must explicitly be True or False; got {value!r}")


def _indexed(rows, key="image_number"):
    numbers = [_integer(row[key], key) for row in rows]
    if len(numbers) != 40 or set(numbers) != set(range(1, 41)):
        raise ValueError(f"Expected exactly 40 unique {key} values covering 1-40")
    return dict(zip(numbers, rows))


def _units(path, count):
    records = _json(path)["result"]["data"]
    result = {str(row["observationUnitDbId"]): row for row in records}
    if len(records) != count or len(result) != count:
        raise ValueError(f"Expected {count} unique saved observation units: {path}")
    return result


def _relationships(unit):
    records = unit["observationUnitPosition"]["observationLevelRelationships"]
    result = {row["levelName"]: _integer(row["levelCode"], "levelCode") for row in records}
    if len(result) != len(records):
        raise ValueError("Duplicate saved observation-level relationship")
    return result


def _source_rows(extension):
    mapping = _indexed(_csv(extension / "bb_records/plant_demo/plant_image_mapping.csv"), "source_image_number")
    plants = _units(extension / "bb_records/plant_demo/observationunits_170_plants_get_verify.json", 40)
    plots = _units(extension / "bb_records/task_c/observationunits_170_get_verify.json", 4)
    rows = []
    for number, original in sorted(mapping.items()):
        row = dict(original, image_number=number, source_image_number=number)
        for field in ("plant_index", "block_number", "replicate_number"):
            row[field] = _integer(row[field], field)
        filename = row["filename"]
        if not filename or Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise ValueError(f"Invalid filename for image {number}")
        image = extension / "demo40/images" / filename
        if not image.is_file():
            raise FileNotFoundError(
                "Download the photographs from Dryad as described in the notebook. "
                f"Place the files listed in data/image_manifest.csv in demo40/images/. Missing: {filename}")
        if (extension / row["image_relative_path"]).resolve() != image.resolve():
            raise ValueError(f"Image path mismatch for image {number}")
        if sha256(image) != row["sha256"]:
            raise ValueError(f"Image identity/SHA-256 mismatch for image {number}")
        if date.fromisoformat(row["capture_date"]).isoformat() != row["capture_date"]:
            raise ValueError(f"Capture date must be YYYY-MM-DD for image {number}")
        plant = plants[row["plant_observationUnitDbId"]]
        plot = plots[row["plot_observationUnitDbId"]]
        if str(plant["additionalInfo"]["observationUnitParent"]) != row["plot_observationUnitDbId"]:
            raise ValueError(f"Parent plot mismatch for image {number}")
        for unit, name_field, level in ((plant, "plant_observationUnitName", "plant"),
                                        (plot, "plot_observationUnitName", "plot")):
            if unit["observationUnitName"] != row[name_field] or unit["observationUnitPosition"]["observationLevel"]["levelName"] != level:
                raise ValueError(f"Saved {level} identity mismatch for image {number}")
            for field in ("studyDbId", "germplasmDbId", "germplasmName"):
                if str(unit[field]) != row[field]:
                    raise ValueError(f"Saved {level} {field} mismatch for image {number}")
            levels = _relationships(unit)
            if levels["block"] != row["block_number"] or levels["rep"] != row["replicate_number"] or row["source_plot_id"] != f"Plot{levels['plot']}":
                raise ValueError(f"Saved {level} block/replicate/plot mismatch for image {number}")
        if row["studyDbId"] != "170" or _relationships(plant)["plant"] != row["plant_index"]:
            raise ValueError(f"Study or plant index mismatch for image {number}")
        # Historical count copies do not participate in the model-to-database flow.
        row.pop("reviewed_count", None)
        row.pop("predicted_count", None)
        row.pop("possible_prediction_truncation", None)
        row["image_path"] = image
        rows.append(row)
    for field in ("filename", "sha256", "plant_observationUnitDbId", "plant_observationUnitName"):
        if len({row[field] for row in rows}) != 40:
            raise ValueError(f"Expected 40 unique {field} values")
    if set(row["plant_observationUnitDbId"] for row in rows) != set(plants):
        raise ValueError("Mapped and saved plant IDs differ")
    if Counter(row["plot_observationUnitDbId"] for row in rows) != Counter({key: 10 for key in plots}):
        raise ValueError("Expected exactly 10 plants in each of four saved plots")
    for plot_id in plots:
        if {row["plant_index"] for row in rows if row["plot_observationUnitDbId"] == plot_id} != set(range(1, 11)):
            raise ValueError(f"Expected unique plant indices 1-10 in plot {plot_id}")
    return rows


def _verify_prediction(item, row):
    number = row["image_number"]
    if _integer(item["image_number"], "image_number") != number or any(item[field] != row[field] for field in ("filename", "sha256")):
        raise ValueError(f"Prediction source identity mismatch for image {number}")
    count = _integer(item["predicted_count"], "predicted_count")
    if _boolean(item["possible_truncation"], "possible_truncation") or count >= SETTINGS["max_det"]:
        raise ValueError(f"Possible prediction truncation for image {number}; stop before upload")
    return count


def load_demo(extension: Path, prediction_run: Path | None = None) -> dict:
    """Read-only local validation. IDs remain strings; ordinal/count fields are integers.

    Prediction counts come exclusively from the selected run, never the mapping's
    historical count copies. No reference workbook or count-review file is read.
    Missing predictions, cap hits, and incomplete runs fail closed.
    No ML imports, model loading, network requests, or filesystem writes occur here.
    """
    extension = Path(extension).resolve()
    if prediction_run is None:
        raise FileNotFoundError(
            "No saved prediction run is bundled. Set run_predictions=True for fresh inference, "
            "or pass prediction_run= with a complete run you previously generated.")
    folder = Path(prediction_run)
    folder = (extension / folder).resolve()
    rows = _source_rows(extension)
    provenance = _json(folder / "provenance.json")
    if provenance["checkpoint_sha256"] != CHECKPOINT_SHA256 or provenance["settings"] != SETTINGS or provenance["versions"] != VERSIONS:
        raise ValueError("Prediction checkpoint/settings/versions differ from the verified combined model run")
    if provenance.get("status", "COMPLETE") != "COMPLETE":
        raise ValueError("Selected prediction run is not complete")
    predictions = _indexed(_csv(folder / "predicted_counts.csv"))
    if {path.name for path in folder.glob("image_*.json")} != {f"image_{number:02}.json" for number in range(1, 41)}:
        raise ValueError("Selected run must have exactly 40 raw image JSON files")
    for row in rows:
        number = row["image_number"]
        count = _verify_prediction(predictions[number], row)
        raw = _json(folder / f"image_{number:02}.json")
        if _verify_prediction(raw, row) != count:
            raise ValueError(f"CSV/raw prediction count mismatch for image {number}")
        if any(len(raw[field]) != count for field in ("classes", "boxes_xyxy", "confidence")) or any(_integer(c, "class") != 0 for c in raw["classes"]):
            raise ValueError(f"Raw Runner detections disagree with count for image {number}")
        if any(not isinstance(c, (int, float)) or not math.isfinite(c) or not SETTINGS["conf"] <= c <= 1 for c in raw["confidence"]):
            raise ValueError(f"Invalid confidence for image {number}")
        if any(len(box) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box) for box in raw["boxes_xyxy"]):
            raise ValueError(f"Invalid detection box for image {number}")
        overlay = folder / "overlays" / f"image_{number:02}.jpg"
        if not overlay.is_file() or overlay.stat().st_size == 0:
            raise ValueError(f"Missing saved prediction overlay for image {number}")
        row.update(predicted_count=count, possible_prediction_truncation=False,
                   prediction_run=folder.name, overlay_path=overlay)
    return dict(rows=rows, provenance=provenance, prediction_dir=folder)


def run_predictions(extension: Path, rows: list) -> Path:
    """Explicit, opt-in GPU operation. Never call during notebook preparation/testing."""
    import importlib.metadata
    import os
    import platform
    import sys
    from uuid import uuid4

    extension = Path(extension).resolve()
    verified_rows = _source_rows(extension)
    supplied = _indexed(rows)
    for row in verified_rows:
        for field, value in row.items():
            if field != "prediction_run" and supplied[row["image_number"]].get(field) != value:
                raise ValueError(f"Input rows changed after verification: image {row['image_number']} {field}")
    if platform.python_version() != "3.10.19":
        raise RuntimeError("New inference requires the verified Python 3.10.19 environment")
    versions = {package: importlib.metadata.version(package) for package in VERSIONS}
    if versions != VERSIONS:
        raise RuntimeError("Installed package versions differ from the verified environment; no installation attempted")
    checkpoint = Path(os.environ.get("STRAWBERRY_RUNNER_CHECKPOINT", str(extension / "models/best.pt"))).expanduser().resolve()
    if not checkpoint.is_file() or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Download the trained weights from this repository's v1.0.0 release and place them at models/best.pt. The checkpoint must match the SHA-256 in MODEL_CARD.md.")
    os.environ.update(YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true")
    for variable, suffix in {"YOLO_CONFIG_DIR": "ultralytics", "MPLCONFIGDIR": "matplotlib", "TORCH_HOME": "torch"}.items():
        os.environ[variable] = str(extension / "cache" / suffix)
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available() or torch.version.cuda != "12.8":
        raise RuntimeError("Verified CUDA 12.8 GPU environment is required; no CPU fallback")
    model = YOLO(str(checkpoint))
    if model.task != "segment" or model.names != {0: "Runner"}:
        raise ValueError("Checkpoint must be the Runner-only segmentation model")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = extension / "demo40/outputs" / f"prediction_{stamp}_{uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "overlays").mkdir()
    provenance = dict(status="INCOMPLETE", checkpoint_sha256=CHECKPOINT_SHA256,
                      settings=SETTINGS, versions=versions, python=sys.executable,
                      device_name=torch.cuda.get_device_name(0), started_utc=stamp)
    (folder / "run_state.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    try:
        outputs = []
        for row in verified_rows:
            if sha256(row["image_path"]) != row["sha256"]:
                raise ValueError(f"Source image changed: {row['filename']}")
            result = model.predict(source=str(row["image_path"]), save=False, verbose=False, **SETTINGS)[0]
            classes = result.boxes.cls.cpu().tolist()
            count = len(classes)
            if any(c != 0 for c in classes) or (count and (result.masks is None or len(result.masks.data) != count)):
                raise ValueError("Inference output must contain matching Runner boxes and masks")
            item = {field: row[field] for field in ("image_number", "filename", "sha256")}
            item.update(predicted_count=count, possible_truncation=count >= SETTINGS["max_det"])
            detail = dict(item, classes=classes, boxes_xyxy=result.boxes.xyxy.cpu().tolist(), confidence=result.boxes.conf.cpu().tolist())
            number = row["image_number"]
            (folder / f"image_{number:02}.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
            result.save(filename=str(folder / "overlays" / f"image_{number:02}.jpg"))
            _verify_prediction(item, row)
            outputs.append(item)
        if len(outputs) != 40 or sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("Incomplete run or checkpoint changed during inference")
        with (folder / "predicted_counts.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(outputs[0]))
            writer.writeheader()
            writer.writerows(outputs)
        provenance.update(status="COMPLETE", image_count=40, finished_utc=datetime.now(timezone.utc).isoformat())
        (folder / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        load_demo(extension, folder)
        (folder / "run_state.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        return folder
    except BaseException as error:
        provenance.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        (folder / "run_state.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        if (folder / "provenance.json").exists():
            (folder / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        raise


# BrAPI payloads, destination checks, and transfer.
"""Local plant-count BrAPI client. Importing this module performs no I/O.

Only model predictions are inputs. Capture dates have day precision and remain
metadata; no capture time or timezone is invented. Uploads default to disabled.
"""
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
import ipaddress
import json
import re
from urllib.parse import urlsplit
STUDY_ID = '170'
VARIABLE_ID = '79806'
VARIABLE_NAME = 'RUNNERNUM_YOLO11x_GI_count_v1|RUNNERDEMO:0000005'
VARIABLE_PUI = 'RUNNERDEMO:0000005'
REFERENCE_SOURCE = 'strawberry-runner-repro/plant-count-v1'
EXPECTED_PLOTS = {'41831': ('UF_GCREC_Runner_2023_Plot1', '41828', 'Florida Brilliance', '1', '1', '1'), '41829': ('UF_GCREC_Runner_2023_Plot2', '41828', 'Florida Brilliance', '1', '1', '2'), '41832': ('UF_GCREC_Runner_2023_Plot3', '41827', 'Florida Medallion', '2', '2', '3'), '41830': ('UF_GCREC_Runner_2023_Plot4', '41827', 'Florida Medallion', '2', '2', '4')}

class ContractError(RuntimeError):
    """The local data or live server does not meet the upload contract."""

def _count(value):
    if isinstance(value, bool):
        raise ContractError('Runner counts must be nonnegative integers, not booleans.')
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ContractError('Runner counts must be nonnegative integers.') from exc
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ContractError('Runner counts must be nonnegative integers.')
    return int(number)

def _brapi_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError('Provenance must contain finite, JSON-compatible values.') from exc

def _rows(rows):
    rows = [dict(row) for row in rows]
    if not rows:
        raise ContractError('The image-to-plant mapping is empty.')
    seen = set()
    for row in rows:
        for field in ('filename', 'sha256', 'capture_date', 'plant_observationUnitDbId', 'plant_observationUnitName', 'plot_observationUnitDbId', 'plot_observationUnitName', 'germplasmDbId', 'germplasmName', 'block_number', 'replicate_number', 'plant_index', 'studyDbId', 'prediction_run'):
            if row.get(field) is None or str(row[field]).strip() == '':
                raise ContractError(f'Mapping field is missing: {field}.')
            row[field] = str(row[field])
        if row['studyDbId'] != STUDY_ID:
            raise ContractError('This demonstration is restricted to study 170.')
        if not re.fullmatch('[0-9a-fA-F]{64}', row['sha256']):
            raise ContractError('Each image needs its SHA256 identity.')
        row['sha256'] = row['sha256'].lower()
        try:
            if date.fromisoformat(row['capture_date']).isoformat() != row['capture_date']:
                raise ValueError
        except ValueError as exc:
            raise ContractError('Capture dates must be YYYY-MM-DD with day precision.') from exc
        row['predicted_count'] = _count(row.get('predicted_count'))
        if str(row.get('possible_prediction_truncation', 'false')).lower() not in ('false', '0', ''):
            raise ContractError('A potentially truncated prediction requires investigation.')
        unit_id = row['plant_observationUnitDbId']
        if not unit_id.isdigit() or unit_id in seen:
            raise ContractError('Plant identifiers must be unique numeric IDs.')
        seen.add(unit_id)
    if len({r['filename'] for r in rows}) != len(rows) or len({r['sha256'] for r in rows}) != len(rows):
        raise ContractError('Each plant must map to a distinct image identity.')
    return rows

def build_payloads(rows, provenance):
    """Build JSON objects from predicted_count, never from manual counts.

    provenance requires checkpoint_sha256; settings and other nonsecret fields
    are preserved inside prediction_provenance. No observationTimeStamp is sent.
    """
    rows = _rows(rows)
    provenance = json.loads(_brapi_json(dict(provenance)))
    checkpoint = str(provenance.get('checkpoint_sha256', '')).lower()
    if not re.fullmatch('[0-9a-f]{64}', checkpoint):
        raise ContractError('Prediction provenance requires checkpoint_sha256.')
    provenance['checkpoint_sha256'] = checkpoint
    payloads = []
    for row in rows:
        info = {'count_source': 'model_prediction', 'capture_date': row['capture_date'], 'capture_date_precision': 'day', 'capture_time_known': False, 'image_filename': row['filename'], 'image_sha256': row['sha256'], 'prediction_run': row['prediction_run'], 'class_id': 0, 'trait_curie': 'CO_372:0000104', 'method_curie': 'RUNNERDEMO_METHOD:0000001', 'scale_curie': 'CO_372:0000086', 'prediction_provenance': provenance}
        identity = {'study': STUDY_ID, 'unit': row['plant_observationUnitDbId'], 'variable': VARIABLE_ID, 'metadata': info}
        reference = 'sha256:' + hashlib.sha256(_brapi_json(identity).encode()).hexdigest()
        payloads.append({'studyDbId': STUDY_ID, 'observationUnitDbId': row['plant_observationUnitDbId'], 'observationVariableDbId': VARIABLE_ID, 'value': str(row['predicted_count']), 'additionalInfo': info, 'externalReferences': [{'referenceID': reference, 'referenceSource': REFERENCE_SOURCE}]})
    return payloads

class PlantBrAPI:

    def __init__(self, base_url='http://localhost:8099/brapi/v2', *, token=None, session=None):
        parsed = urlsplit(base_url)
        host = parsed.hostname
        try:
            loopback = host == 'localhost' or ipaddress.ip_address(host or '').is_loopback
        except ValueError:
            loopback = False
        if parsed.scheme not in ('http', 'https') or not loopback or parsed.username or parsed.password or parsed.query or parsed.fragment or (parsed.path.rstrip('/') != '/brapi/v2'):
            raise ContractError('Only a loopback /brapi/v2 base URL is permitted.')
        if token is not None and (not isinstance(token, str) or any((c.isspace() for c in token))):
            raise ContractError('The access token must be a single non-whitespace value.')
        if session is None:
            import requests
            session = requests.Session()
        if hasattr(session, 'trust_env'):
            session.trust_env = False
        self.base_url = base_url.rstrip('/')
        self._token = token
        self._session = session
        self._verified_unit_ids = None

    def _request(self, method, path, *, params=None, payload=None):
        if method not in ('GET', 'POST') or not re.fullmatch('[a-z]+(?:/\\d+)?', path):
            raise ContractError('Unsupported API operation.')
        headers = {'Accept': 'application/json'}
        if self._token:
            headers['Authorization'] = 'Bearer ' + self._token
        if payload is not None:
            headers['Content-Type'] = 'application/json'
        try:
            response = self._session.request(method, self.base_url + '/' + path, params=params, json=payload, headers=headers, timeout=(5, 30), allow_redirects=False)
        except Exception as exc:
            raise ContractError('API request failed; verify server state before retrying a write.') from None
        if not 200 <= response.status_code < 300:
            raise ContractError(f'API returned HTTP {response.status_code}; redirects and automatic write retries are disabled.')
        try:
            body = response.json()
        except (ValueError, TypeError):
            raise ContractError('API response was not valid JSON.') from None
        if not isinstance(body, dict) or not isinstance(body.get('metadata'), dict) or 'result' not in body:
            raise ContractError('API response is missing its BrAPI envelope.')
        for item in body['metadata'].get('status', []):
            if not isinstance(item, dict):
                raise ContractError('Malformed API status entry.')
            if str(item.get('messageType', '')).upper() in ('ERROR', 'FATAL') or any((str(key).isdigit() and int(key) >= 400 for key in item)):
                raise ContractError('The API reported an error; response body is withheld from logs.')
        return body

    def get_all(self, resource, **filters):
        if resource not in ('observationunits', 'observations'):
            raise ContractError('Unsupported paginated resource.')
        (records, seen_pages) = ([], set())
        for page in range(1000):
            body = self._request('GET', resource, params={**filters, 'page': page, 'pageSize': 1000})
            result = body['result']
            pagination = body['metadata'].get('pagination', {})
            if not isinstance(result, dict) or not isinstance(result.get('data'), list):
                raise ContractError('API list result is malformed.')
            data = result['data']
            try:
                total = int(pagination['totalCount'])
                current = int(pagination['currentPage'])
            except (KeyError, ValueError, TypeError):
                raise ContractError('API pagination is missing or invalid.') from None
            if current != page or total < 0:
                raise ContractError('API pagination did not advance correctly.')
            signature = _brapi_json(data)
            if data and signature in seen_pages:
                raise ContractError('API repeated a page; refusing incomplete duplicate checks.')
            seen_pages.add(signature)
            records.extend(data)
            if len(records) == total:
                return records
            if not data or len(records) > total:
                raise ContractError('API counts and returned records disagree.')
        raise ContractError('API pagination exceeded the bounded page limit.')

    def preflight(self, rows):
        rows = _rows(rows)
        if len(rows) != 40 or Counter((r['plot_observationUnitDbId'] for r in rows)) != Counter({p: 10 for p in EXPECTED_PLOTS}):
            raise ContractError('This mapping must contain forty plants, ten in each of the four verified plots.')
        study = self._request('GET', 'studies/' + STUDY_ID)['result']
        variable = self._request('GET', 'variables/' + VARIABLE_ID)['result']
        if str(study.get('studyDbId')) != STUDY_ID or str(study.get('locationDbId')) != '28' or str(study.get('additionalInfo', {}).get('programDbId')) != '169' or (study.get('experimentalDesign', {}).get('PUI') != 'RCBD') or ('2023' not in [str(x) for x in study.get('seasons', [])]):
            raise ContractError('Live study identity differs from the approved photographed subset.')
        if str(variable.get('observationVariableDbId')) != VARIABLE_ID or variable.get('observationVariableName') != VARIABLE_NAME or variable.get('observationVariablePUI') != VARIABLE_PUI or (variable.get('scale', {}).get('dataType') != 'Numerical'):
            raise ContractError('Live runner-variable identity or numeric scale changed.')
        units = self.get_all('observationunits', studyDbId=STUDY_ID, observationUnitLevelName='plant')
        by_id = {str(unit.get('observationUnitDbId')): unit for unit in units}
        if len(units) != 40 or len(by_id) != 40 or set(by_id) != {r['plant_observationUnitDbId'] for r in rows}:
            raise ContractError('Live plant records differ from the approved forty-image mapping.')
        for row in rows:
            unit = by_id[row['plant_observationUnitDbId']]
            parent = row['plot_observationUnitDbId']
            (name, accession, accession_name, block, rep, number) = EXPECTED_PLOTS[parent]
            position = unit.get('observationUnitPosition', {})
            relationships = {x.get('levelName'): str(x.get('levelCode')) for x in position.get('observationLevelRelationships', [])}
            checks = [str(unit.get('studyDbId')) == STUDY_ID, unit.get('observationUnitName') == row['plant_observationUnitName'], row['plant_observationUnitName'] == name + '_plant_' + row['plant_index'], row['plant_index'] in {str(x) for x in range(1, 11)}, str(unit.get('additionalInfo', {}).get('observationUnitParent')) == parent, row['plot_observationUnitName'] == name, str(unit.get('germplasmDbId')) == row['germplasmDbId'] == accession, unit.get('germplasmName') == row['germplasmName'] == accession_name, relationships.get('block') == row['block_number'] == block, relationships.get('rep') == row['replicate_number'] == rep, relationships.get('plot') == number, position.get('observationLevel', {}).get('levelName') == 'plant', str(position.get('observationLevel', {}).get('levelCode')) == row['plant_index']]
            if not all(checks):
                raise ContractError(f"Live parent, genotype, or design differs for plant {row['plant_observationUnitDbId']}.")
        self._verified_unit_ids = set(by_id)
        return {'status': 'PREFLIGHT_VERIFIED', 'studyDbId': STUDY_ID, 'plant_count': 40, 'variableDbId': VARIABLE_ID, 'ontology_components': 'Trait, method, and scale identifiers are pinned in this helper; flat BrAPI variable response omits contains links.'}

    def _existing(self):
        units = self.get_all('observationunits', studyDbId=STUDY_ID, observationUnitLevelName='plant', includeObservations='true')
        unit_ids = {str(unit.get('observationUnitDbId')) for unit in units}
        if len(units) != 40 or len(unit_ids) != 40:
            raise ContractError('Nested readback must include all forty distinct plants.')
        if self._verified_unit_ids is not None and unit_ids != self._verified_unit_ids:
            raise ContractError('Plant membership changed after preflight.')
        observations = []
        for unit in units:
            if str(unit.get('studyDbId')) != STUDY_ID or unit.get('observationUnitPosition', {}).get('observationLevel', {}).get('levelName') != 'plant':
                raise ContractError('Nested readback contains an unexpected study or observation level.')
            if not isinstance(unit.get('observations'), list):
                raise ContractError('Server did not include observations; duplicate status is unknown.')
            for observation in unit['observations']:
                if str(observation.get('observationVariableDbId')) == VARIABLE_ID:
                    if str(observation.get('studyDbId')) != STUDY_ID or str(observation.get('observationUnitDbId')) != str(unit.get('observationUnitDbId')):
                        raise ContractError('Nested observation identity is inconsistent.')
                    observations.append(observation)
        return observations

    def classify(self, payloads):
        existing = defaultdict(list)
        for observation in self._existing():
            existing[str(observation['observationUnitDbId'])].append(observation)
        result = {'new': [], 'duplicates': [], 'conflicts': []}
        for payload in payloads:
            matches = existing[str(payload['observationUnitDbId'])]
            if not matches:
                result['new'].append(payload)
            elif len(matches) == 1 and _count(matches[0].get('value')) == _count(payload['value']) and (matches[0].get('additionalInfo') == payload['additionalInfo']) and (matches[0].get('externalReferences') == payload['externalReferences']):
                result['duplicates'].append(matches[0])
            else:
                result['conflicts'].append({'plant_id': payload['observationUnitDbId'], 'existing_observation_ids': [x.get('observationDbId') for x in matches], 'reason': 'An observation already exists with different or incomplete value/provenance.'})
        result.update({'status': 'CONFLICT' if result['conflicts'] else 'CLASSIFIED', 'missing_count': len(result['new']), 'identical_count': len(result['duplicates']), 'conflict_count': len(result['conflicts']), 'readback_verified': not result['new'] and (not result['conflicts'])})
        return result

    def upload(self, payloads, rows, *, enabled=False):
        payloads = json.loads(_brapi_json(list(payloads)))
        if enabled is not True:
            return {'status': 'UPLOAD_DISABLED', 'planned_count': len(payloads), 'posted_count': 0, 'readback_verified': False}
        if not self._token:
            raise ContractError('An in-memory access token is required for an explicitly enabled upload.')
        rows = _rows(rows)
        if not payloads or len(payloads) != len(rows):
            raise ContractError('Payload and mapping lengths differ.')
        by_id = {r['plant_observationUnitDbId']: r for r in rows}
        if len({p.get('observationUnitDbId') for p in payloads}) != len(rows):
            raise ContractError('Payload plant IDs are not unique.')
        for payload in payloads:
            row = by_id.get(str(payload.get('observationUnitDbId')))
            if row is None or set(payload) != {'studyDbId', 'observationUnitDbId', 'observationVariableDbId', 'value', 'additionalInfo', 'externalReferences'}:
                raise ContractError('Payload identities/fields differ from the approved contract; no updates or timestamps are permitted.')
            if payload['studyDbId'] != STUDY_ID or payload['observationVariableDbId'] != VARIABLE_ID:
                raise ContractError('Payload study or variable differs from the approved contract.')
            info = payload['additionalInfo']
            rebuilt = build_payloads([row], info.get('prediction_provenance', {}))[0]
            if payload != rebuilt:
                raise ContractError('Payload differs from the model prediction or recorded provenance.')
        self.preflight(rows)
        classification = self.classify(payloads)
        if classification['conflicts']:
            raise ContractError('Existing observations conflict with this batch. Nothing was posted; overwrites are not supported.')
        new = classification['new']
        if new:
            self._request('POST', 'observations', payload=new)
        verified = self.classify(payloads)
        if verified['new'] or verified['conflicts']:
            raise ContractError('Upload readback is incomplete or conflicting. Do not retry automatically; inspect server records.')
        return {'status': 'UPLOAD_VERIFIED' if new else 'ALREADY_PRESENT', 'posted_count': len(new), 'duplicate_count': len(classification['duplicates']), 'identical_count': len(verified['duplicates']), 'missing_count': 0, 'conflict_count': 0, 'readback_verified': True, 'verified_observation_ids': [r['observationDbId'] for r in verified['duplicates']]}

SELECTED_PHOTOGRAPHS = ['cropped_20231128_R1B2_west_102_21-14R-119_png_jpg.rf.b7e3618d2d41cae63336dfbdd953614a.jpg', 'cropped_20231128_R1B2_west_106_21-14R-190_png_jpg.rf.e594ef0baebf3a3752a0243ddb8a1d8c.jpg', 'cropped_20231128_R1B2_west_114_22-84-279_png_jpg.rf.ec395fbbadf91e59006abfb93949f221.jpg']
FLOWCHART_SVG = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<svg xmlns="http://www.w3.org/2000/svg" width="1036" height="496" viewBox="0 0 1036 496" role="img" aria-labelledby="flow-title flow-desc">\n  <title id="flow-title">From plant photographs to breeding records</title>\n  <desc id="flow-desc">Six numbered steps connect 40 ground photographs to model-predicted runner counts, a predefined standardized trait mapping, BrAPI transfer, local Breedbase plant records, and read-back confirmation. Model training was completed earlier in the linked repository. The notebook settings determine whether the workflow uses saved predictions or performs live inference and transfer.</desc>\n  <defs>\n    <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">\n      <path d="M 0 0 L 10 5 L 0 10 z" fill="#618274" />\n    </marker>\n  </defs>\n  <rect width="1036" height="496" fill="#FFFFFF" />\n  <g font-family="Segoe UI, Arial, sans-serif" fill="#203A30">\n    <text x="32" y="32" font-size="25" font-weight="600">From plant photographs to breeding records</text>\n    <text x="32" y="55" font-size="14" fill="#5F7369">A runner-count demonstration for breeders and agricultural researchers</text>\n    <a href="https://github.com/Bidhan-ghimire/strawberry-runner-repro" target="_blank">\n      <rect x="347" y="78" width="342" height="68" rx="10" fill="#F6F8F7" stroke="#ADB8B1" stroke-width="1.3" stroke-dasharray="5 4" />\n      <text x="518" y="102" text-anchor="middle" font-size="16" font-weight="600" fill="#53675C">Model training completed earlier</text>\n      <text x="518" y="124" text-anchor="middle" font-size="13" fill="#53675C">Outside this demo · open the training repository</text>\n    </a>\n    <path d="M 518 146 L 518 166" fill="none" stroke="#618274" stroke-width="1.8" stroke-dasharray="4 3" marker-end="url(#arrow)" />\n    <path d="M 328 229 L 360 229" fill="none" stroke="#618274" stroke-width="2" marker-end="url(#arrow)" />\n    <path d="M 666 229 L 698 229" fill="none" stroke="#618274" stroke-width="2" marker-end="url(#arrow)" />\n    <path d="M 856 282 L 856 316" fill="none" stroke="#618274" stroke-width="2" marker-end="url(#arrow)" />\n    <path d="M 708 379 L 676 379" fill="none" stroke="#618274" stroke-width="2" marker-end="url(#arrow)" />\n    <path d="M 370 379 L 338 379" fill="none" stroke="#618274" stroke-width="2" marker-end="url(#arrow)" />\n    <g id="step-photos">\n      <rect x="32" y="176" width="296" height="106" rx="12" fill="#F0F7EF" stroke="#AAC3A2" stroke-width="1.4" />\n      <circle cx="55" cy="203" r="12" fill="#416E59" />\n      <text x="55" y="208" text-anchor="middle" font-size="14" font-weight="600" fill="#FFFFFF">1</text>\n      <text x="76" y="208" font-size="17" font-weight="600">Plant photographs</text>\n      <text x="50" y="237" font-size="14.5" fill="#415B4E">40 ground images</text>\n      <text x="50" y="259" font-size="14.5" fill="#415B4E">One photographed plant per record</text>\n    </g>\n    <g id="step-model">\n      <rect x="370" y="176" width="296" height="106" rx="12" fill="#E1EFE5" stroke="#96B5A1" stroke-width="1.4" />\n      <circle cx="393" cy="203" r="12" fill="#416E59" />\n      <text x="393" y="208" text-anchor="middle" font-size="14" font-weight="600" fill="#FFFFFF">2</text>\n      <text x="414" y="208" font-size="17" font-weight="600">Trained YOLO11x model</text>\n      <text x="388" y="237" font-size="14.5" fill="#415B4E">Detect visible runners</text>\n      <text x="388" y="259" font-size="14.5" fill="#415B4E">Predict a count for each plant image</text>\n    </g>\n    <g id="step-ontology">\n      <rect x="708" y="176" width="296" height="106" rx="12" fill="#E6F2EF" stroke="#9DBDB4" stroke-width="1.4" />\n      <circle cx="731" cy="203" r="12" fill="#416E59" />\n      <text x="731" y="208" text-anchor="middle" font-size="14" font-weight="600" fill="#FFFFFF">3</text>\n      <text x="752" y="208" font-size="17" font-weight="600">Standardized trait mapping</text>\n      <text x="726" y="237" font-size="14.5" fill="#415B4E">Runner number + method</text>\n      <text x="726" y="259" font-size="14.5" fill="#415B4E">+ count scale</text>\n    </g>\n    <g id="step-brapi">\n      <rect x="708" y="326" width="296" height="106" rx="12" fill="#EDF6F5" stroke="#9FBDB8" stroke-width="1.4" />\n      <circle cx="731" cy="353" r="12" fill="#416E59" />\n      <text x="731" y="358" text-anchor="middle" font-size="14" font-weight="600" fill="#FFFFFF">4</text>\n      <text x="752" y="358" font-size="17" font-weight="600">Transfer through BrAPI</text>\n      <text x="726" y="387" font-size="14.5" fill="#415B4E">Send each predicted count</text>\n      <text x="726" y="409" font-size="14.5" fill="#415B4E">with its plant and capture date</text>\n    </g>\n    <g id="step-breedbase">\n      <rect x="370" y="326" width="296" height="106" rx="12" fill="#DAEEE5" stroke="#93B4A3" stroke-width="1.4" />\n      <circle cx="393" cy="353" r="12" fill="#416E59" />\n      <text x="393" y="358" text-anchor="middle" font-size="14" font-weight="600" fill="#FFFFFF">5</text>\n      <text x="414" y="358" font-size="17" font-weight="600">Local Breedbase</text>\n      <text x="388" y="387" font-size="14.5" fill="#415B4E">Individual plant records</text>\n      <text x="388" y="409" font-size="14.5" fill="#415B4E">linked to plots and accessions</text>\n    </g>\n    <g id="step-verify">\n      <rect x="32" y="326" width="296" height="106" rx="12" fill="#F0F7EF" stroke="#AAC3A2" stroke-width="1.4" />\n      <circle cx="55" cy="353" r="12" fill="#416E59" />\n      <text x="55" y="358" text-anchor="middle" font-size="14" font-weight="600" fill="#FFFFFF">6</text>\n      <text x="76" y="358" font-size="17" font-weight="600">Read back and confirm</text>\n      <text x="50" y="387" font-size="14.5" fill="#415B4E">Retrieve the saved measurements</text>\n      <text x="50" y="409" font-size="14.5" fill="#415B4E">Check plant, date, trait, and count</text>\n    </g>\n    <line x1="32" y1="453" x2="1004" y2="453" stroke="#D7E3DA" />\n    <text x="518" y="478" text-anchor="middle" font-size="14" fill="#5F7369">Image → model count → ontology mapping → BrAPI → plant record → read-back.</text>\n  </g>\n</svg>'

# Public functions used by the reader-facing flow below.
def _NOTICE(title, message, color="#23654b"):
    display(HTML(f'<div style="border-left:5px solid {color};background:#f3f7f4;'
                 f'padding:14px 18px;margin:12px 0"><strong>{escape(title)}</strong>'
                 f'<div>{escape(message)}</div></div>'))

def _LOAD_IMAGES():
    global rows, frame
    if not rows:
        rows = _source_rows(EXTENSION)
        frame = pd.DataFrame(rows)

def _SELECTED_ROWS():
    _LOAD_IMAGES()
    by_name = {row["filename"]: row for row in rows}
    return [by_name[name] for name in SELECTED_PHOTOGRAPHS]

def SHOW_PIPELINE_FLOWCHART():
    display(SVG(FLOWCHART_SVG))

def SHOW_STRAWBERRY_IMAGES():
    examples = _SELECTED_ROWS()
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=100)
    for axis, row in zip(axes, examples):
        with Image.open(row["image_path"]) as image:
            axis.imshow(image)
        axis.set_title(f"Ground image {row['image_number']}", fontsize=12)
        axis.axis("off")
    figure.tight_layout()
    plt.show()
    plt.close(figure)

def COUNT_STRAWBERRY_RUNNER():
    global demo, rows, frame, payloads, upload_result, retrieval_result, client
    _LOAD_IMAGES()
    if RUN_PREDICTIONS:
        folder = run_predictions(EXTENSION, rows)
        demo = load_demo(EXTENSION, folder)
        rows = demo["rows"]
        mode = "New model predictions"
    else:
        demo = load_demo(EXTENSION, PREDICTION_RUN)
        rows = demo["rows"]
        mode = "Saved model predictions — no new inference"
    # A different run must not leave stale mapped/uploaded values in kernel memory.
    frame = pd.DataFrame(rows)
    payloads, upload_result, retrieval_result, client = [], None, None, None
    _NOTICE(mode, f"Runner counts available for all {len(rows)} plant images.")
    for row in _SELECTED_ROWS():
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.3), dpi=100)
        with Image.open(row["image_path"]) as image:
            axes[0].imshow(image)
        with Image.open(row["overlay_path"]) as overlay:
            axes[1].imshow(overlay)
        axes[0].set_title(f"Ground image {row['image_number']}")
        axes[1].set_title(f"Predicted runners: {row['predicted_count']}")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        plt.show()
        plt.close(figure)
    counts = frame[["image_number", "predicted_count"]].rename(
        columns={"image_number": "Image", "predicted_count": "Predicted runners"})
    display(HTML('<details><summary>See counts for all 40 images</summary>'
                 + counts.to_html(index=False, escape=True, border=0) + '</details>'))

def MAP_RUNNER_COUNTS_TO_ONTOLOGY():
    global payloads
    _LOAD_IMAGES()
    if demo is None:
        raise RuntimeError("Run COUNT_STRAWBERRY_RUNNER() before mapping the measurements.")
    payloads = build_payloads(rows, demo["provenance"])
    if len(payloads) != 40:
        raise RuntimeError("Expected one measurement for each of 40 photographed plants.")
    mapping = pd.DataFrame([
        ["Trait", "Runner number", "CO_372:0000104"],
        ["Method", "YOLO11x GI runner counting v1 (local)", "RUNNERDEMO_METHOD:0000001"],
        ["Scale", "count", "CO_372:0000086"],
        ["Breedbase variable", "RUNNERNUM_YOLO11x_GI_count_v1 (local)", "79806"],
    ], columns=["Component", "Meaning", "Identifier"])
    display(HTML(mapping.to_html(index=False, escape=True, border=0)))
    _NOTICE("40 predictions mapped", "Each model count is associated with the defined runner trait, method, and scale.")

def _CLIENT():
    return PlantBrAPI(BRAPI_BASE, token=os.environ.get("BREEDBASE_BRAPI_TOKEN"))

def UPLOAD_RUNNER_COUNTS_TO_BREEDBASE():
    global client, upload_result
    if not payloads:
        raise RuntimeError("Run MAP_RUNNER_COUNTS_TO_ONTOLOGY() before the upload step.")
    preview = pd.DataFrame(rows)[["image_number", "plant_observationUnitDbId",
                                  "capture_date", "predicted_count"]].copy()
    preview.columns = ["Image", "Breedbase plant ID", "Photographed", "Count to send"]
    display(HTML('<details open><summary>Measurements prepared for individual plants</summary>'
                 + preview.to_html(index=False, escape=True, border=0) + '</details>'))
    if not DO_UPLOAD:
        upload_result = {"status": "UPLOAD_DISABLED", "readback_verified": False}
        _NOTICE("Preview only — upload is disabled", "No database request was made by this step.")
        return
    client = _CLIENT()
    upload_result = client.upload(payloads, rows, enabled=True)
    _NOTICE("Upload result: " + upload_result["status"],
            f"{upload_result['posted_count']} new measurements; {upload_result['duplicate_count']} already present; "
            f"{len(upload_result['verified_observation_ids'])} records verified by read-back.")

def VERIFY_STORED_RUNNER_COUNTS():
    global client, retrieval_result
    if not payloads:
        raise RuntimeError("Map the predictions before checking the stored records.")
    if DO_RETRIEVE:
        client = _CLIENT()
        client.preflight(rows)
        retrieval_result = client.classify(payloads)
        complete = retrieval_result["readback_verified"]
        _NOTICE("All 40 stored measurements match" if complete else "Live read-back is incomplete",
                f"{retrieval_result['identical_count']} match; {retrieval_result['missing_count']} missing; "
                f"{retrieval_result['conflict_count']} conflicts.", color="#23654b" if complete else "#96671f")
    elif upload_result and upload_result.get("readback_verified"):
        _NOTICE("Upload already verified by a fresh read-back", "All 40 stored measurements matched the model values and provenance.")
    else:
        _NOTICE("Live storage not checked in this preview",
                "Live retrieval is disabled. Prepared measurements are not proof of a database upload.", color="#96671f")

def SHOW_REPRODUCIBILITY_DETAILS():
    _LOAD_IMAGES()
    details = {
        "Study / plant records": "170 / 40 plants, ten per parent plot",
        "Observation variable": "79806 — RUNNERNUM_YOLO11x_GI_count_v1",
        "Saved prediction run": demo["prediction_dir"].name,
        "Checkpoint SHA-256": demo["provenance"]["checkpoint_sha256"],
        "Model settings": json.dumps(demo["provenance"]["settings"], sort_keys=True),
        "Capture date": "Day precision, retained as provenance; no capture time invented",
        "Plant IDs": "Local sequential IDs; not historical field tags or spatial positions",
        "Repeated measurements": "Outside this one-image, one-measurement-per-plant demonstration",
        "Source date note": "Dryad reports 2023-11-29; the owner's corrected local mapping uses 2023-11-28 for Plot1–Plot3. The difference is unresolved; local dates were preserved.",
        "Actions enabled": f"New inference: {RUN_PREDICTIONS}; upload: {DO_UPLOAD}; live retrieval: {DO_RETRIEVE}",
    }
    detail_table = pd.DataFrame(details.items(), columns=["Item", "Value"])
    filenames = pd.DataFrame(rows)[["image_number", "filename", "sha256", "plant_observationUnitName"]]
    display(HTML('<details><summary>Settings and source identifiers</summary>'
                 + detail_table.to_html(index=False, escape=True, border=0)
                 + filenames.to_html(index=False, escape=True, border=0) + '</details>'))

def SETUP_STRAWBERRY_RUNNER_PIPELINE(*, run_predictions=False, upload=False,
                                    retrieve=False, prediction_run=None):
    """Initialize the switches; inference or an explicit local run is required for counts."""
    global RUN_PREDICTIONS, DO_UPLOAD, DO_RETRIEVE, PREDICTION_RUN
    global demo, rows, frame, payloads, upload_result, retrieval_result, client
    if any(type(value) is not bool for value in (run_predictions, upload, retrieve)):
        raise TypeError("Pipeline switches must be True or False.")
    RUN_PREDICTIONS, DO_UPLOAD, DO_RETRIEVE = run_predictions, upload, retrieve
    PREDICTION_RUN = prediction_run
    demo, rows, frame, payloads = None, [], None, []
    upload_result = retrieval_result = client = None
    _NOTICE("Notebook functions ready",
            f"New inference: {RUN_PREDICTIONS}; upload: {DO_UPLOAD}; live retrieval: {DO_RETRIEVE}.")

__all__ = ["SETUP_STRAWBERRY_RUNNER_PIPELINE", "SHOW_PIPELINE_FLOWCHART",
           "SHOW_STRAWBERRY_IMAGES", "COUNT_STRAWBERRY_RUNNER",
           "MAP_RUNNER_COUNTS_TO_ONTOLOGY", "UPLOAD_RUNNER_COUNTS_TO_BREEDBASE",
           "VERIFY_STORED_RUNNER_COUNTS", "SHOW_REPRODUCIBILITY_DETAILS"]
