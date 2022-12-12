from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os


# The DAG object; we'll need thid to instantiate a DAG
from airflow import DAG
training_pipeline = None
# Operators; we need this to operate!
from airflow.operators.python import PythonOperator

# [END imporETL DAG tutorial_prediction',
# [START default_args]
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
with DAG(
        'yolov7',
        default_args={'retries': 2},
        # [END default_args]
        description='Object Tracking with YOLOv7 and DeepSort',
        schedule_interval="@weekly",
        start_date=pendulum.datetime(2022, 11, 12, tz="UTC"),
        catchup=False,
        tags=['example'],
) as dag:
    # [END instantiate_dag]
    # [START documentation]
    dag.doc_md = __doc__
    # [END documentation]

    # [START extract_function]

    from src.pipeline.tracking import TrackingPipeline

    tracking_pipeline = TrackingPipeline()


    def model_ingestion(**kwargs):
        ti = kwargs['ti']
        model_ingestion_artifacts = tracking_pipeline.start_model_ingestion()
        print(model_ingestion_artifacts)
        ti.xcom_push('model_ingestion_artifacts', model_ingestion_artifacts.to_dict())

    def model_loading(**kwargs):
        from src.entity.artifact_entity import ModelIngestionArtifacts
        ti = kwargs['ti']
        model_ingestion_artifacts = ti.xcom_pull(task_ids="model_ingestion", key="model_ingestion_artifacts")
        model_ingestion_artifacts = ModelIngestionArtifacts(**(model_ingestion_artifacts))
        print(model_ingestion_artifacts)
        model_loading_artifacts = tracking_pipeline.start_model_loading(
            model_ingestion_artifacts=model_ingestion_artifacts
        )
        ti.xcom_push('model_loading_artifacts', model_loading_artifacts.to_dict())


    def data_transformation(**kwargs):
        from src.entity.artifact_entity import ModelLoadingArtifacts
        ti = kwargs['ti']
        model_loading_artifacts = ti.xcom_pull(task_ids="model_loading", key="model_loading_artifacts")
        model_loading_artifacts = ModelLoadingArtifacts(**(model_loading_artifacts))

        data_transformation_artifacts = tracking_pipeline.start_data_transformation(
            model_loading_artifacts=model_loading_artifacts
        )
        ti.xcom_push('data_transformation_artifacts', data_transformation_artifacts.to_dict())

    def object_tracking(**kwargs):
        from src.entity.artifact_entity import DataTransformationArtifacts, ModelLoadingArtifacts
        ti = kwargs['ti']
        data_transformation_artifacts = ti.xcom_pull(task_ids="data_transformation", key="data_transformation_artifacts")
        data_transformation_artifacts = DataTransformationArtifacts(**(data_transformation_artifacts))

        model_loading_artifacts = ti.xcom_pull(task_ids="model_loading", key="model_loading_artifacts")
        model_loading_artifacts = ModelLoadingArtifacts(*(model_loading_artifacts))

        object_tracking_artifacts = tracking_pipeline.start_object_tracking(
            data_transformation_artifacts=data_transformation_artifacts,
            model_loading_artifacts=model_loading_artifacts
        )
        ti.xcom_push('object_tracking_artifacts', object_tracking_artifacts.to_dict())

    def pusher(**kwargs):
        ti = kwargs['ti']
        object_tracking_artifacts = ti.xcom_pull(task_ids="pusher", key="object_tracking_artifacts")

        pusher_artifacts = tracking_pipeline.start_pusher()
        print(f'Model pusher artifacts: {pusher_artifacts}')
        ti.xcom_push('object_tracking_artifacts', object_tracking_artifacts.to_dict())

        print("Tracking pipeline completed video saved to GCloud Storage")


    model_ingestion = PythonOperator(
        task_id='model_ingestion',
        python_callable=model_ingestion,
    )
    model_ingestion.doc_md = dedent(
        """\
    #### Extract task
    A simple Extract task to get data ready for the rest of the data pipeline.
    In this case, getting data is simulated by reading from a hardcoded JSON string.
    This data is then put into xcom, so that it can be processed by the next task.
    """
    )
    model_loading = PythonOperator(
        task_id="model_loading",
        python_callable=model_loading
    )

    data_transformation = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformation
    )

    object_tracking = PythonOperator(
        task_id="object_tracking",
        python_callable=object_tracking

    )
    pusher = PythonOperator(
        task_id="pusher",
        python_callable=pusher

    )

    model_ingestion >> model_loading >> data_transformation >> object_tracking >> pusher
