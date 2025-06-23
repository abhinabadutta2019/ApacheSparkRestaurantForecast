package org.example;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.ml.*;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;

import static org.apache.spark.sql.functions.*;

public class GBTOptimizedFinal {
    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder()
                .appName("GBTOptimizedFinal")
                .master("local[*]")
                .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/db_spark.collection1")
                .getOrCreate();

        Dataset<Row> df = spark.read()
                .format("mongodb")
                .option("uri", "mongodb://127.0.0.1/db_spark.collection1")
                .load();

        Dataset<Row> filtered = df.na().drop()
                .filter("Revenue > 0 AND Revenue < 500000")
                .withColumn("Rating_x_Budget", col("Rating").multiply(col("Marketing Budget")))
                .withColumn("Ambience_x_Chef", col("Ambience Score").multiply(col("Chef Experience Years")))
                .withColumn("RevenueLog", log1p(col("Revenue")));

        String[] catCols = {"Cuisine", "Location", "Parking Availability"};
        String[] numCols = {
                "Rating", "Marketing Budget", "Number of Reviews",
                "Average Meal Price", "Ambience Score", "Chef Experience Years",
                "Service Quality Score", "Social Media Followers",
                "Rating_x_Budget", "Ambience_x_Chef"
        };

        String[] indexCols = new String[catCols.length];
        String[] oneHotCols = new String[catCols.length];
        StringIndexer[] indexers = new StringIndexer[catCols.length];
        OneHotEncoder[] encoders = new OneHotEncoder[catCols.length];

        for (int i = 0; i < catCols.length; i++) {
            indexCols[i] = catCols[i] + "_Index";
            oneHotCols[i] = catCols[i] + "_Vec";
            indexers[i] = new StringIndexer().setInputCol(catCols[i]).setOutputCol(indexCols[i]);
            encoders[i] = new OneHotEncoder().setInputCol(indexCols[i]).setOutputCol(oneHotCols[i]);
        }

        String[] allFeatures = new String[oneHotCols.length + numCols.length];
        System.arraycopy(oneHotCols, 0, allFeatures, 0, oneHotCols.length);
        System.arraycopy(numCols, 0, allFeatures, oneHotCols.length, numCols.length);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(allFeatures)
                .setOutputCol("features");

        GBTRegressor gbt = new GBTRegressor()
                .setLabelCol("RevenueLog")
                .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                indexers[0], indexers[1], indexers[2],
                encoders[0], encoders[1], encoders[2],
                assembler, gbt
        });

        Dataset<Row>[] splits = filtered.randomSplit(new double[]{0.8, 0.2}, 42);
        Dataset<Row> trainData = splits[0].cache();
        Dataset<Row> testData = splits[1];

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(gbt.maxDepth(), new int[]{3})
                .addGrid(gbt.maxIter(), new int[]{40})
                .build();

        RegressionEvaluator logEval = new RegressionEvaluator()
                .setLabelCol("RevenueLog")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        TrainValidationSplit tvs = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(logEval)
                .setEstimatorParamMaps(paramGrid)
                .setTrainRatio(0.8);

        long start = System.currentTimeMillis();
        TrainValidationSplitModel model = tvs.fit(trainData);
        Dataset<Row> predictions = model.transform(testData);
        long end = System.currentTimeMillis();
        double duration = (end - start) / 1000.0;

        predictions = predictions.withColumn("prediction_actual", expm1(col("prediction")));

        double rmseLog = logEval.evaluate(predictions);
        System.out.println("Log-scale RMSE: " + rmseLog);
        System.out.println("Time taken: " + duration + " seconds");

        RegressionEvaluator realEval = new RegressionEvaluator()
                .setLabelCol("Revenue")
                .setPredictionCol("prediction_actual")
                .setMetricName("rmse");

        double realRmse = realEval.evaluate(predictions);
        System.out.println("Actual RMSE (Revenue scale): " + realRmse);

        Dataset<Row> mapeDF = predictions.withColumn(
                "abs_percent_error",
                abs(col("Revenue").minus(col("prediction_actual")))
                        .divide(col("Revenue"))
                        .multiply(100)
        );

        Row mapeRow = mapeDF.agg(avg("abs_percent_error").alias("MAPE")).first();
        double mape = mapeRow.getDouble(0);
        double accuracy = 100.0 - mape;

        System.out.println("MAPE (%): " + mape);
        System.out.println("Estimated Accuracy (%): " + accuracy);

        RegressionEvaluator r2Eval = new RegressionEvaluator()
                .setLabelCol("Revenue")
                .setPredictionCol("prediction_actual")
                .setMetricName("r2");

        double r2 = r2Eval.evaluate(predictions);
        System.out.println("R² Score: " + r2);

        // ✅ NEW: Save predictions to separate MongoDB collection
        predictions.select("Name", "Revenue", "prediction_actual")
                .write()
                .format("mongodb")
                .mode("append")
                .option("database", "db_spark")
                .option("collection", "predictions_gbt")
                .option("uri", "mongodb://127.0.0.1")
                .save();


        spark.stop();
    }
}
