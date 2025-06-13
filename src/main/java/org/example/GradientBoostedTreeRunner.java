package org.example;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.ml.regression.GBTRegressionModel;

import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.expressions.Window;

public class GradientBoostedTreeRunner {
    public static void main(String[] args) {

        // 1. Create Spark session
        SparkSession spark = SparkSession.builder()
                .appName("MongoDBSparkGBT")
                .master("local[*]")
                .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/db_spark.collection1")
                .getOrCreate();

        // 2. Load data from MongoDB
        Dataset<Row> df = spark.read()
                .format("mongodb")
                .option("uri", "mongodb://127.0.0.1/db_spark.collection1")
                .load();

        // 3. Prepare Data
        Dataset<Row> dfWithName = df.select(
                "Name", "Rating", "Marketing Budget", "Number of Reviews",
                "Average Meal Price", "Ambience Score", "Chef Experience Years", "Revenue");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[] {
                        "Rating", "Marketing Budget", "Number of Reviews",
                        "Average Meal Price", "Ambience Score", "Chef Experience Years"
                })
                .setOutputCol("features");

        Dataset<Row> assembledData = assembler.transform(dfWithName)
                .select("Name", "features", "Revenue")
                .na().drop();

        // 4. Train Gradient Boosted Tree Model
        GBTRegressor gbt = new GBTRegressor()
                .setLabelCol("Revenue")
                .setFeaturesCol("features")
                .setMaxIter(50); // You can tune this

        GBTRegressionModel gbtModel = gbt.fit(assembledData);
        Dataset<Row> predictions = gbtModel.transform(assembledData);

        // 5. Evaluate RMSE
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("Revenue")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(predictions);
        System.out.println("GBT Regressor RMSE: " + rmse);

        // 6. Calculate MAPE and Accuracy
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

        // 7. R² Score
        RegressionEvaluator r2Eval = new RegressionEvaluator()
                .setLabelCol("Revenue")
                .setPredictionCol("prediction")
                .setMetricName("r2");

        double r2 = r2Eval.evaluate(predictions);
        System.out.println("R² Score: " + r2);

        // 8. Show sample predictions
        System.out.println("Sample Predictions:");
        predictions.select("Name", "Revenue", "prediction").show(5);

        // 9. Save predictions to CSV
        predictions.select("Name", "Revenue", "prediction")
                .coalesce(1)
                .write()
                .mode("overwrite")
                .option("header", "true")
                .csv("/home/abhinaba/Downloads/Codes/RestaurantForecast/output/gbt_predictions");

        // 10. Stop Spark
        spark.stop();
    }
}
