package org.example;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;

import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.expressions.Window;

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

        // 6. Evaluate model (RMSE)
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("Revenue")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(predictions);
        System.out.println("Random Forest RMSE: " + rmse);

        // 7. MAPE and Accuracy
        Dataset<Row> mapeDF = predictions.withColumn(
                "abs_percent_error",
                abs(col("Revenue").minus(col("prediction")))
                        .divide(col("Revenue"))
                        .multiply(100)
        );

        Row mapeRow = mapeDF.agg(avg("abs_percent_error").alias("MAPE")).first();
        double mape = mapeRow.getDouble(0);
        double accuracy = 100.0 - mape;

        System.out.println("MAPE (%): " + mape);
        System.out.println("Estimated Accuracy (%): " + accuracy);

        // 8. R² Score
        RegressionEvaluator r2Eval = new RegressionEvaluator()
                .setLabelCol("Revenue")
                .setPredictionCol("prediction")
                .setMetricName("r2");

        double r2 = r2Eval.evaluate(predictions);
        System.out.println("R² Score: " + r2);

        // 9. Show sample output
        System.out.println("Sample Predictions:");
        predictions.select("Name", "Revenue", "prediction").show(5);

        // 10. Save predictions to CSV
        predictions.select("Name", "Revenue", "prediction")
                .coalesce(1)
                .write()
                .mode("overwrite")
                .option("header", "true")
                .csv("/home/abhinaba/Downloads/Codes/RestaurantForecast/output/rf_predictions");

        // 11. Stop Spark session
        spark.stop();
    }
}
