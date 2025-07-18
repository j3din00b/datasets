import json
import tarfile

import pytest

from datasets import Audio, DownloadManager, Features, Image, List, Value
from datasets.packaged_modules.webdataset.webdataset import WebDataset

from ..utils import (
    require_numpy1_on_windows,
    require_pil,
    require_sndfile,
    require_torch,
    require_torchcodec,
)


@pytest.fixture
def gzipped_text_wds_file(tmp_path, text_gz_path):
    filename = tmp_path / "file.tar"
    num_examples = 3
    with tarfile.open(str(filename), "w") as f:
        for example_idx in range(num_examples):
            f.add(text_gz_path, f"{example_idx:05d}.txt.gz")
    return str(filename)


@pytest.fixture
def image_wds_file(tmp_path, image_file):
    json_file = tmp_path / "data.json"
    filename = tmp_path / "file.tar"
    num_examples = 3
    with json_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"caption": "this is an image"}))
    with tarfile.open(str(filename), "w") as f:
        for example_idx in range(num_examples):
            f.add(json_file, f"{example_idx:05d}.json")
            f.add(image_file, f"{example_idx:05d}.jpg")
    return str(filename)


@pytest.fixture
def audio_wds_file(tmp_path, audio_file):
    json_file = tmp_path / "data.json"
    filename = tmp_path / "file.tar"
    num_examples = 3
    with json_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"transcript": "this is a transcript"}))
    with tarfile.open(str(filename), "w") as f:
        for example_idx in range(num_examples):
            f.add(json_file, f"{example_idx:05d}.json")
            f.add(audio_file, f"{example_idx:05d}.wav")
    return str(filename)


@pytest.fixture
def bad_wds_file(tmp_path, image_file, text_file):
    json_file = tmp_path / "data.json"
    filename = tmp_path / "bad_file.tar"
    with json_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"caption": "this is an image"}))
    with tarfile.open(str(filename), "w") as f:
        f.add(image_file)
        f.add(json_file)
    return str(filename)


@pytest.fixture
def tensor_wds_file(tmp_path, tensor_file):
    json_file = tmp_path / "data.json"
    filename = tmp_path / "file.tar"
    num_examples = 3
    with json_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "this is a text"}))
    with tarfile.open(str(filename), "w") as f:
        for example_idx in range(num_examples):
            f.add(json_file, f"{example_idx:05d}.json")
            f.add(tensor_file, f"{example_idx:05d}.pth")
    return str(filename)


@require_pil
def test_gzipped_text_webdataset(gzipped_text_wds_file, text_path):
    data_files = {"train": [gzipped_text_wds_file]}
    webdataset = WebDataset(data_files=data_files)
    split_generators = webdataset._split_generators(DownloadManager())
    assert webdataset.info.features == Features(
        {
            "__key__": Value("string"),
            "__url__": Value("string"),
            "txt.gz": Value("string"),
        }
    )
    assert len(split_generators) == 1
    split_generator = split_generators[0]
    assert split_generator.name == "train"
    generator = webdataset._generate_examples(**split_generator.gen_kwargs)
    _, examples = zip(*generator)
    assert len(examples) == 3
    assert isinstance(examples[0]["txt.gz"], str)
    with open(text_path, "r") as f:
        assert examples[0]["txt.gz"].replace("\r\n", "\n") == f.read().replace("\r\n", "\n")


@require_pil
def test_image_webdataset(image_wds_file):
    import PIL.Image

    data_files = {"train": [image_wds_file]}
    webdataset = WebDataset(data_files=data_files)
    split_generators = webdataset._split_generators(DownloadManager())
    assert webdataset.info.features == Features(
        {
            "__key__": Value("string"),
            "__url__": Value("string"),
            "json": {"caption": Value("string")},
            "jpg": Image(),
        }
    )
    assert len(split_generators) == 1
    split_generator = split_generators[0]
    assert split_generator.name == "train"
    generator = webdataset._generate_examples(**split_generator.gen_kwargs)
    _, examples = zip(*generator)
    assert len(examples) == 3
    assert isinstance(examples[0]["json"], dict)
    assert isinstance(examples[0]["json"]["caption"], str)
    assert isinstance(examples[0]["jpg"], dict)  # keep encoded to avoid unecessary copies
    encoded = webdataset.info.features.encode_example(examples[0])
    decoded = webdataset.info.features.decode_example(encoded)
    assert isinstance(decoded["json"], dict)
    assert isinstance(decoded["json"]["caption"], str)
    assert isinstance(decoded["jpg"], PIL.Image.Image)


@require_pil
def test_image_webdataset_missing_keys(image_wds_file):
    import PIL.Image

    data_files = {"train": [image_wds_file]}
    features = Features(
        {
            "__key__": Value("string"),
            "__url__": Value("string"),
            "json": {"caption": Value("string")},
            "jpg": Image(),
            "jpeg": Image(),  # additional field
            "txt": Value("string"),  # additional field
        }
    )
    webdataset = WebDataset(data_files=data_files, features=features)
    split_generators = webdataset._split_generators(DownloadManager())
    assert webdataset.info.features == features
    split_generator = split_generators[0]
    assert split_generator.name == "train"
    generator = webdataset._generate_examples(**split_generator.gen_kwargs)
    _, example = next(iter(generator))
    encoded = webdataset.info.features.encode_example(example)
    decoded = webdataset.info.features.decode_example(encoded)
    assert isinstance(decoded["json"], dict)
    assert isinstance(decoded["json"]["caption"], str)
    assert isinstance(decoded["jpg"], PIL.Image.Image)
    assert decoded["jpeg"] is None
    assert decoded["txt"] is None


@require_torchcodec
@require_sndfile
def test_audio_webdataset(audio_wds_file):
    from torchcodec.decoders import AudioDecoder

    data_files = {"train": [audio_wds_file]}
    webdataset = WebDataset(data_files=data_files)
    split_generators = webdataset._split_generators(DownloadManager())
    assert webdataset.info.features == Features(
        {
            "__key__": Value("string"),
            "__url__": Value("string"),
            "json": {"transcript": Value("string")},
            "wav": Audio(),
        }
    )
    assert len(split_generators) == 1
    split_generator = split_generators[0]
    assert split_generator.name == "train"
    generator = webdataset._generate_examples(**split_generator.gen_kwargs)
    _, examples = zip(*generator)
    assert len(examples) == 3
    assert isinstance(examples[0]["json"], dict)
    assert isinstance(examples[0]["json"]["transcript"], str)
    assert isinstance(examples[0]["wav"], dict)
    assert isinstance(examples[0]["wav"]["bytes"], bytes)  # keep encoded to avoid unecessary copies
    encoded = webdataset.info.features.encode_example(examples[0])
    decoded = webdataset.info.features.decode_example(encoded)
    assert isinstance(decoded["json"], dict)
    assert isinstance(decoded["json"]["transcript"], str)
    assert isinstance(decoded["wav"], AudioDecoder)


def test_webdataset_errors_on_bad_file(bad_wds_file):
    data_files = {"train": [bad_wds_file]}
    webdataset = WebDataset(data_files=data_files)
    with pytest.raises(ValueError):
        webdataset._split_generators(DownloadManager())


@require_pil
def test_webdataset_with_features(image_wds_file):
    import PIL.Image

    data_files = {"train": [image_wds_file]}
    features = Features(
        {
            "__key__": Value("string"),
            "__url__": Value("string"),
            "json": {"caption": Value("string"), "additional_field": Value("int64")},
            "jpg": Image(),
        }
    )
    webdataset = WebDataset(data_files=data_files, features=features)
    split_generators = webdataset._split_generators(DownloadManager())
    assert webdataset.info.features == features
    split_generator = split_generators[0]
    assert split_generator.name == "train"
    generator = webdataset._generate_examples(**split_generator.gen_kwargs)
    _, example = next(iter(generator))
    encoded = webdataset.info.features.encode_example(example)
    decoded = webdataset.info.features.decode_example(encoded)
    assert decoded["json"]["additional_field"] is None
    assert isinstance(decoded["json"], dict)
    assert isinstance(decoded["json"]["caption"], str)
    assert isinstance(decoded["jpg"], PIL.Image.Image)


@require_numpy1_on_windows
@require_torch
def test_tensor_webdataset(tensor_wds_file):
    import torch

    data_files = {"train": [tensor_wds_file]}
    webdataset = WebDataset(data_files=data_files)
    split_generators = webdataset._split_generators(DownloadManager())
    assert webdataset.info.features == Features(
        {
            "__key__": Value("string"),
            "__url__": Value("string"),
            "json": {"text": Value("string")},
            "pth": List(Value("float32")),
        }
    )
    assert len(split_generators) == 1
    split_generator = split_generators[0]
    assert split_generator.name == "train"
    generator = webdataset._generate_examples(**split_generator.gen_kwargs)
    _, examples = zip(*generator)
    assert len(examples) == 3
    assert isinstance(examples[0]["json"], dict)
    assert isinstance(examples[0]["json"]["text"], str)
    assert isinstance(examples[0]["pth"], torch.Tensor)  # keep encoded to avoid unecessary copies
    encoded = webdataset.info.features.encode_example(examples[0])
    decoded = webdataset.info.features.decode_example(encoded)
    assert isinstance(decoded["json"], dict)
    assert isinstance(decoded["json"]["text"], str)
    assert isinstance(decoded["pth"], list)
