import sys
from src.logger import logging
from src.exception import CustomException
from src.configuration.gcloud_syncer import GCloudSync
from src.entity.config_entity import PusherConfig
from src.entity.artifact_entity import PusherArtifacts, ObjectTrackingArtifacts


class Pusher:
    def __init__(self, pusher_config: PusherConfig,
                 object_tracking_artifacts: ObjectTrackingArtifacts):
        """
        :param pusher_config: Configuration for artifacts pusher
        :param object_tracking_artifacts: Artifacts of Object tracking
        """
        self.pusher_config = pusher_config
        self.object_tracking_artifacts = object_tracking_artifacts
        self.gcloud = GCloudSync()

    def initiate_pusher(self) -> PusherArtifacts:
        """
            Method Name :   initiate_pusher
            Description :   This method initiates pusher.

            Output      :    Pusher artifact
        """
        logging.info("Entered initiate_pusher method of PusherArtifacts class")
        try:
            self.gcloud.sync_folder_to_gcloud(self.pusher_config.BUCKET_NAME,
                                              self.object_tracking_artifacts.artifacts_path,
                                              self.pusher_config.DIR_NAME)

            logging.info("Uploaded tracked video to gcloud storage")

            logging.info("Saving the pusher artifacts")
            pusher_artifact = PusherArtifacts(
                bucket_name=self.pusher_config.BUCKET_NAME,
                dir_name=self.pusher_config.DIR_NAME
            )
            logging.info("Exited the initiate_pusher method of Pusher class")
            return pusher_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
