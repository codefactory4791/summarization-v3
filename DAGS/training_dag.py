from datetime import datetime, timedelta
from airflow import DAG
# from airflow.contrib.sensors.gcs_sensors import GoogleCloudStorageObjectSensor
from airflow.hooks.base_hook import BaseHook
#from airflow.operators.app_engine_admin_plugin import AppEngineVersionOperator
from airflow.operators.ml_engine_plugin import MLEngineTrainingOperator
#from airflow.models import Variable


PROJECT_ID = "call-summarizatiion"

# GCS bucket names and region, can also be changed.
BUCKET = 'gs://us-east1-composer-training-50d83041-bucket'
OBJECT = 'data/train.csv'
TRAINING_FILE = BUCKET + OBJECT
REGION = 'us-east1' #'europe-west2' #'us-east1'
JOB_ID = 'T5_Train_{0}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M'))
JOB_DIR = BUCKET + '/jobs/' + JOB_ID
OUTPUT_DIR = "summarization_bucket_2023/model_artifacts/Training_Job"


master_image_uri='gcr.io/call-summarizatiion/training-image:v1'
MASTER_CONFIG = {"imageUri": master_image_uri,}

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email": ["designingprodmlsystems23@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True
 
}

#DATAFLOW_PARAMS = Variable.get("dataflow_parasms", deserialize_json = True)

# the arguments used by the train.py file

TRAINING_ARGS = ['--job-dir', JOB_DIR,
                 '--train-file', TRAINING_FILE,
                 '--output-dir', OUTPUT_DIR
                 ]      

dag = DAG(

    'training_dag',
    defaukt_args = DEFAULT_ARGS,
    schedule_interval = '@hourly'
)
# trigger = GoogleCloudStorageObjectSensor(

#     task_id = 'File_Validation',
#     bucket = BUCKET,
#     object = OBJECT,
#     google_cloud_conn_id='google_cloud_default',
#     timeout=1000,
#     dag=dag
# )


model_training = MLEngineTrainingOperator(

    task_id='Training_Job',
    project_id=PROJECT_ID,
    job_id=JOB_ID,
    training_args=TRAINING_ARGS,
    region=REGION,
    scale_tier='CUSTOM',
    master_type='complex_model_m_gpu',
    master_config=MASTER_CONFIG,
    dag=dag


)

model_training
# trigger >> model_training