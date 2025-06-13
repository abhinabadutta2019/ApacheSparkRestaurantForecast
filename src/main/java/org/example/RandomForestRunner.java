package org.example;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;

public class RandomForestRunner {
    public static void main(String[] args) {

        // 1. Create Spark session
        SparkSession spark = SparkSession.builder()
                .appName("MongoDBSparkRF")
                .master("local[*]")
                .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/db_spark.collection1")
                .getOrCreate();

        // 2. Load data
        Dataset<Row> df = spark.read()
                .format("mongodb")
                .option("uri", "mongodb://127.0.0.1/db_spark.collection1")
                .load();

        // 3. Select required columns
        Dataset<Row> dfWithName = df.select(
                "Name", "Rating", "Marketing Budget", "Number of Reviews",
                "Average Meal Price", "Ambience Score", "Chef Experience Years", "Revenue");

        // 4. Assemble features
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[] {
                        "Rating", "Marketing Budget", "Number of Reviews",
                        "Average Meal Price", "Ambience Score", "Chef Experience Years"
                })
                .setOutputCol("features");

        Dataset<Row> assembledData = assembler.transform(dfWithName)
                .select("Name", "features", "Revenue")
                .na().drop();

        // 5. Train Random Forest model
        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol("Revenue")
                .setFeaturesCol("features");

        RandomForestRegressionModel model = rf.fit(assembledData);
        Dataset<Row> predictions = model.transform(assembledData);

        // 6. Evaluate model
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("Revenue")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(predictions);
        System.out.println("Random Forest RMSE: " + rmse);

        // 7. Show sample output
        predictions.select("Name", "Revenue", "prediction").show(5);

        // 8. Save predictions to CSV
        predictions.select("Name", "Revenue", "prediction")
                .coalesce(1)
                .write()
                .mode("overwrite")
                .option("header", "true")
                .csv("/home/abhinaba/Downloads/Codes/RestaurantForecast/output/rf_predictions");

        // 9. Stop Spark session
        spark.stop();
    }
}
