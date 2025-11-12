// ========================================================
//  Sistem Deteksi Hoax Berita
//  Dibuat oleh Fina – 2025
// ========================================================

using Microsoft.ML;
using Microsoft.ML.Data;
using System;

public class BeritaData
{
    [LoadColumn(0)]
    public string Berita { get; set; }

    [LoadColumn(1)]
    public string Label { get; set; }
}

public class BeritaPrediction
{
    [ColumnName("PredictedLabel")]
    public string Prediksi { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("========================================");
        Console.WriteLine("   Sistem Deteksi Hoax Berita");
        Console.WriteLine("   Dibuat oleh Fina");
        Console.WriteLine("========================================\n");

        var mlContext = new MLContext();
        string dataPath = "fina_dataset_hoax_valid.csv";


        // 1. Load dataset
        Console.WriteLine("Memuat dataset...");
        var data = mlContext.Data.LoadFromTextFile<BeritaData>(
            dataPath,
            separatorChar: ',',
            hasHeader: true
        );

        // 2. Split train-test
        var split = mlContext.Data.TrainTestSplit(data, 0.2);

        // 3. Pipeline machine learning
        var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(BeritaData.Berita))
            .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // 4. Train model
        Console.WriteLine("Training model...");
        var model = pipeline.Fit(split.TrainSet);

        // 5. Evaluate
        Console.WriteLine("Evaluasi model...");
        var predictions = model.Transform(split.TestSet);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

        Console.WriteLine($"\nAkurasi Micro : {metrics.MicroAccuracy}");
        Console.WriteLine($"Akurasi Macro : {metrics.MacroAccuracy}\n");

        // 6. Save model
        mlContext.Model.Save(model, split.TrainSet.Schema, "HoaxDetikModel.zip");
        Console.WriteLine("Model berhasil disimpan sebagai 'HoaxDetikModel.zip'\n");

        // 7. Prediksi manual
        Console.WriteLine("Coba deteksi berita manual!");
        Console.Write("Masukkan teks berita: ");
        string inputText = Console.ReadLine();

        var engine = mlContext.Model.CreatePredictionEngine<BeritaData, BeritaPrediction>(model);

        var input = new BeritaData { Berita = inputText };
        var result = engine.Predict(input);

        Console.WriteLine($"\nHasil Prediksi → {result.Prediksi.ToUpper()}");
        Console.WriteLine("\nTerima kasih telah menggunakan aplikasi ini!");
    }
}
