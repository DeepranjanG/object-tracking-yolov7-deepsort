FROM google/cloud-sdk:latest

RUN apt-get update -y \
&& apt-get install -y software-properties-common \
&& add-apt-repository ppa:deadsnakes/ppa \
&& apt-get install python3-pip -y \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

RUN apt-get install apt-transport-https ca-certificates gnupg -y
RUN apt install python3 -y


ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

RUN mkdir /app
COPY . /app/
WORKDIR /app/
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu &&  \
    pip3 install -r requirements.txt

RUN airflow db init
RUN airflow users create  -e deepranjan@ineuron.ai -f Deepranjan -l Gupta -p admin -r Admin  -u admin
RUN chmod 777 start.sh
ENTRYPOINT [ "/bin/sh" ]
CMD ["start.sh"]