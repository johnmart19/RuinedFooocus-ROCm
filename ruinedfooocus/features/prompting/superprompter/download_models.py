from huggingface_hub import snapshot_download
import os


def download_models(modelDir):
    model_name = "roborovski/superprompt-v1"
    from ruinedfooocus.app.arguments import args

    snapshot_download(
        model_name,
        local_dir=modelDir,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.txt"],
        local_files_only=args.offline or os.environ.get("RF_OFFLINE") == "1",
    )
    print("Downloaded SuperPrompt-v1 model files to", modelDir)
    return modelDir


if __name__ == "__main__":
    from ruinedfooocus.core.paths import data_path

    download_models(data_path("superprompter", "model_files"))
