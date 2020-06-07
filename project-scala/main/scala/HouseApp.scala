import org.apache.spark._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object HouseApp{
	def main(args: Array[String]) {

		var file = "dataset/housing.csv"

	    val spark = SparkSession.builder.appName("House Price").getOrCreate()

		var df = spark.read.format("csv").option("inferSchema", "true").option("header", true).load(file)

		df.show(4)

		df.describe("housing_median_age","total_rooms","population","median_house_value").show

		df = df.withColumn("rooms_per_house", col("total_rooms")/col("households"))
		df = df.withColumn("pop_per_house", col("population")/col("households"))
		df = df.withColumn("bedrooms_per_room", col("total_bedrooms")/col("total_rooms"))

		df=df.drop("total_rooms","households", "population" , "totalbedrooms")

		df.show(4)

		val indexer = new StringIndexer().setInputCol("ocean_proximity")
		                                 .setOutputCol("ocean_proximity_in")

		var new_df = indexer.fit(df).transform(df)

		new_df=new_df.drop("ocean_proximity")

		new_df.show(4)

		new_df.cache()
		new_df.createOrReplaceTempView("house")

		new_df.select(corr("median_house_value", "median_income")).show()

		new_df.select(corr("median_house_value", "bedrooms_per_room")).show()

		new_df.select(corr("median_house_value", "pop_per_house")).show()

		new_df.select(corr("median_house_value", "ocean_proximity_in")).show()

		val Array(trainingData, testData) = new_df.randomSplit(Array(0.8, 0.2), 1234)

		new_df.columns

		val featureCols = Array("housing_median_age", "median_income", "rooms_per_house", "pop_per_house",
		                        "bedrooms_per_room", "longitude", "latitude", "ocean_proximity_in")

		val assembler = new VectorAssembler().setHandleInvalid("skip")
		                                     .setInputCols(featureCols)
		                                     .setOutputCol("rawfeatures")

		val scaler = new StandardScaler().setInputCol("rawfeatures")
		                                 .setOutputCol("features")

		val rf = new RandomForestRegressor().setLabelCol("median_house_value")
		                                    .setFeaturesCol("features")

		val steps = Array(assembler, scaler, rf)

		val pipeline = new Pipeline().setStages(steps)

		val paramGrid = new ParamGridBuilder().addGrid(rf.maxBins, Array(50, 100))
		                                      .addGrid(rf.maxDepth, Array(7, 10, 20))
		                                      .addGrid(rf.numTrees, Array(20, 40))
		                                      .build()


		val evaluator = new RegressionEvaluator().setLabelCol("median_house_value")
		                                         .setPredictionCol("prediction")
		                                         .setMetricName("rmse")

		val crossvalidator = new CrossValidator().setEstimator(pipeline)
		                                         .setEvaluator(evaluator)
		                                         .setEstimatorParamMaps(paramGrid)
		                                         .setNumFolds(3)

		val pipelineModel = crossvalidator.fit(trainingData)

		val featureImportances = pipelineModel.bestModel
		                                      .asInstanceOf[PipelineModel]
		                                      .stages(2)
		                                      .asInstanceOf[RandomForestRegressionModel]
		                                      .featureImportances

		assembler.getInputCols.zip(featureImportances.toArray)
		                      .sortBy(-_._2)
		                      .foreach { case (feat, imp) => println(s"feature: $feat, importance: $imp") }

		val bestEstimatorParamMap = pipelineModel.getEstimatorParamMaps
		                                         .zip(pipelineModel.avgMetrics)
		                                         .maxBy(_._2)
		                                         ._1
		println(s"Best params:\n$bestEstimatorParamMap")

		val predictions = pipelineModel.transform(testData)

		predictions.select("prediction", "median_house_value").show(5)

		val predictions_error = predictions.withColumn("error", col("prediction")-col("median_house_value"))
		predictions_error.select("prediction", "median_house_value", "error").show

		predictions_error.describe("prediction", "median_house_value", "error").show

		val maevaluator = new RegressionEvaluator().setLabelCol("median_house_value")
		                                           .setMetricName("mae")
		val mae = maevaluator.evaluate(predictions)

		val evaluator = new RegressionEvaluator().setLabelCol("median_house_value")
		                                         .setMetricName("rmse")
		val rmse = evaluator.evaluate(predictions)

		pipelineModel.write.overwrite().save("modeldir")

		val sameModel = CrossValidatorModel.load("modeldir")

	    spark.stop()
	}
}
