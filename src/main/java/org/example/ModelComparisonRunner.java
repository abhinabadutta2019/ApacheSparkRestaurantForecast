package org.example;

import org.apache.spark.sql.*;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.GBTRegressor;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.OneHotEncoder;

import java.util.*;

import static org.apache.spark.sql.functions.*;

public class ModelComparisonRunner {
    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder()
                .appName("Compare LinearRegression, RandomForest and GBT")
                .master("local[*]")
                .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/db_spark.collection1")
                .getOrCreate();

        Dataset<Row> df = spark.read()
                .format("mongodb")
                .option("uri", "mongodb://127.0.0.1/db_spark.collection1")
                .load()
                .na().drop()
                .filter("Revenue > 0 AND Revenue < 500000");

        System.out.println("\n=== Sample Cleaned Input Data ===");
        df.show(10);

        String[] numCols = {
                "Rating", "Marketing Budget", "Number of Reviews",
                "Average Meal Price", "Ambience Score", "Chef Experience Years"
        };

        // LINEAR REGRESSION
        Dataset<Row> dfLR = df.select(
                col("Name"), col("Revenue"),
                col("Rating"), col("Marketing Budget"), col("Number of Reviews"),
                col("Average Meal Price"), col("Ambience Score"), col("Chef Experience Years")
        );

        VectorAssembler assemblerLR = new VectorAssembler().setInputCols(numCols).setOutputCol("features");
        Dataset<Row> assembledLR = assemblerLR.transform(dfLR).select("Name", "features", "Revenue");

        LinearRegression lr = new LinearRegression().setLabelCol("Revenue").setFeaturesCol("features");
        Dataset<Row> predictionsLR = lr.fit(assembledLR).transform(assembledLR);

        predictionsLR.select("Revenue", "prediction").show(10);

        double rmseLR = new RegressionEvaluator().setLabelCol("Revenue").setPredictionCol("prediction").setMetricName("rmse").evaluate(predictionsLR);
        double r2LR = new RegressionEvaluator().setLabelCol("Revenue").setPredictionCol("prediction").setMetricName("r2").evaluate(predictionsLR);
        double mapeLR = predictionsLR.withColumn("abs_percent_error", abs(col("Revenue").minus(col("prediction"))).divide(col("Revenue")).multiply(100)).agg(avg("abs_percent_error")).first().getDouble(0);
        double accLR = 100.0 - mapeLR;

        // RANDOM FOREST
        RandomForestRegressor rf = new RandomForestRegressor().setLabelCol("Revenue").setFeaturesCol("features");
        Dataset<Row> predictionsRF = rf.fit(assembledLR).transform(assembledLR);

        predictionsRF.select("Revenue", "prediction").show(10);

        double rmseRF = new RegressionEvaluator().setLabelCol("Revenue").setPredictionCol("prediction").setMetricName("rmse").evaluate(predictionsRF);
        double r2RF = new RegressionEvaluator().setLabelCol("Revenue").setPredictionCol("prediction").setMetricName("r2").evaluate(predictionsRF);
        double mapeRF = predictionsRF.withColumn("abs_percent_error", abs(col("Revenue").minus(col("prediction"))).divide(col("Revenue")).multiply(100)).agg(avg("abs_percent_error")).first().getDouble(0);
        double accRF = 100.0 - mapeRF;

        // GBT
        Dataset<Row> dfGBT = df.withColumn("Rating_x_Budget", col("Rating").multiply(col("Marketing Budget")))
                .withColumn("Ambience_x_Chef", col("Ambience Score").multiply(col("Chef Experience Years")))
                .withColumn("RevenueLog", log1p(col("Revenue")));

        String[] fullNumCols = Arrays.copyOf(numCols, numCols.length + 2);
        fullNumCols[numCols.length] = "Rating_x_Budget";
        fullNumCols[numCols.length + 1] = "Ambience_x_Chef";

        String[] catCols = {"Cuisine", "Location", "Parking Availability"};
        String[] oneHotCols = new String[catCols.length];
        StringIndexer[] indexers = new StringIndexer[catCols.length];
        OneHotEncoder[] encoders = new OneHotEncoder[catCols.length];

        for (int i = 0; i < catCols.length; i++) {
            String indexed = catCols[i] + "_Index";
            String vec = catCols[i] + "_Vec";
            indexers[i] = new StringIndexer().setInputCol(catCols[i]).setOutputCol(indexed);
            encoders[i] = new OneHotEncoder().setInputCol(indexed).setOutputCol(vec);
            oneHotCols[i] = vec;
        }

        String[] allGBTFeatures = Arrays.copyOf(oneHotCols, oneHotCols.length + fullNumCols.length);
        System.arraycopy(fullNumCols, 0, allGBTFeatures, oneHotCols.length, fullNumCols.length);

        VectorAssembler assemblerGBT = new VectorAssembler().setInputCols(allGBTFeatures).setOutputCol("features");

        GBTRegressor gbt = new GBTRegressor()
                .setLabelCol("RevenueLog")
                .setFeaturesCol("features")
                .setMaxDepth(3)
                .setMaxIter(40);

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                indexers[0], indexers[1], indexers[2],
                encoders[0], encoders[1], encoders[2],
                assemblerGBT, gbt
        });

        Dataset<Row>[] gbtSplit = dfGBT.randomSplit(new double[]{0.8, 0.2}, 42);
        Dataset<Row> trainGBT = gbtSplit[0];
        Dataset<Row> testGBT = gbtSplit[1];

        Dataset<Row> predictionsGBT = pipeline.fit(trainGBT).transform(testGBT)
                .withColumn("prediction_actual", expm1(col("prediction")));

        predictionsGBT.select("Revenue", "prediction_actual").show(10);

        double rmseGBT = new RegressionEvaluator().setLabelCol("Revenue").setPredictionCol("prediction_actual").setMetricName("rmse").evaluate(predictionsGBT);
        double r2GBT = new RegressionEvaluator().setLabelCol("Revenue").setPredictionCol("prediction_actual").setMetricName("r2").evaluate(predictionsGBT);
        double mapeGBT = predictionsGBT.withColumn("abs_percent_error", abs(col("Revenue").minus(col("prediction_actual"))).divide(col("Revenue")).multiply(100)).agg(avg("abs_percent_error")).first().getDouble(0);
        double accGBT = 100.0 - mapeGBT;

        // Summary Table
        System.out.println("\n=== Model Comparison Summary ===");
        System.out.printf("%-20s %-12s %-12s %-14s %-12s%n", "Model", "RMSE", "MAPE (%)", "Accuracy (%)", "R² Score");
        System.out.printf("%-20s %-12.2f %-12.2f %-14.2f %-12.4f%n", "Linear Regression", rmseLR, mapeLR, accLR, r2LR);
        System.out.printf("%-20s %-12.2f %-12.2f %-14.2f %-12.4f%n", "Random Forest", rmseRF, mapeRF, accRF, r2RF);
        System.out.printf("%-20s %-12.2f %-12.2f %-14.2f %-12.4f%n", "GBT Optimized", rmseGBT, mapeGBT, accGBT, r2GBT);

        spark.stop();
    }
}
