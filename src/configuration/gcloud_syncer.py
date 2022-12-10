import os


class GCloudSync:

    def sync_folder_to_gcloud(self, gcp_bucket_url, filepath, foldername):

        command = f"gsutil -m cp -r {filepath} gs://{gcp_bucket_url}/{foldername}/"
        # command = f"gcloud storage cp {filepath}/{filename} gs://{gcp_bucket_url}/"
        os.system(command)

    def sync_folder_from_gcloud(self, gcp_bucket_url, foldername, destination):

        command = f"gsutil -m cp -r gs://{gcp_bucket_url}/{foldername} {destination}/"
        # command = f"gcloud storage cp gs://{gcp_bucket_url}/{filename} {destination}/{filename}"
        os.system(command)
