# Predicting Housing Prices using Apache Spark (Scala)

## Directory Notebook
- Jupyter notebook with scala code to make ETL and Machine Learning

## Directory project-scala
- With code to submit in ```spark-submit```

## Directory modeldir
- Where was saved training model logs and parameters

---
### Submit scala job with spark-submit in local machine:

1. In directory ```project-scala```:
```bash
sbt package
```

2. Submit job
```bash
spark-submit \
--class "HouseApp" \
--master local \
target/scala-2.11/house-project_2.11-1.0.jar   