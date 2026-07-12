package com.example.skindiseasecheck;

import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.text.Html;


import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.text.HtmlCompat;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    private ImageView imageView;
    private Button btnSelectImage;
    private TextView resultTextView;

    private Bitmap selectedBitmap;
    private Interpreter tflite;
    private final String[] classLabels = {
            "Eczema", "Warts/Viral Infections", "Melanoma", "Atopic Dermatitis",
            "BCC", "Melanocytic Nevi", "Benign Lesions", "Psoriasis/Lichen Planus",
            "Seborrheic Keratoses", "Fungal Infections"
    };

    private ActivityResultLauncher<String> imagePickerLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle(getString(R.string.about_title));

        View dialogView = getLayoutInflater().inflate(R.layout.about_dialog, null);
        TextView aboutTextView = dialogView.findViewById(R.id.aboutTextView);


        //  Ensures line breaks and emojis display correctly

        aboutTextView.setText(HtmlCompat.fromHtml(getString(R.string.about_description), HtmlCompat.FROM_HTML_MODE_LEGACY));



        builder.setView(dialogView);
        builder.setPositiveButton("OK", null);
        builder.show();




        imageView = findViewById(R.id.imageView);
        btnSelectImage = findViewById(R.id.selectImageBtn);
        resultTextView = findViewById(R.id.predictionText);

        // Load TFLite model
        try {
            tflite = new Interpreter(FileUtil.loadMappedFile(this, "skin_disease_model.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
            resultTextView.setText("Error loading model." + e.getMessage());
        }

        // Register image picker launcher
        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
                        try {
                            selectedBitmap = getBitmapFromUri(uri);
                            imageView.setImageBitmap(selectedBitmap);
                            runModel(selectedBitmap);  // Run model immediately
                        } catch (IOException e) {
                            e.printStackTrace();
                            resultTextView.setText("Failed to load image.");
                        }
                    }
                });

        btnSelectImage.setOnClickListener(v -> imagePickerLauncher.launch("image/*"));
    }

    private Bitmap getBitmapFromUri(Uri uri) throws IOException {
        Bitmap originalBitmap;

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            ImageDecoder.Source source = ImageDecoder.createSource(getContentResolver(), uri);
            originalBitmap = ImageDecoder.decodeBitmap(source);
        } else {
            originalBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
        }

        //  Convert to mutable bitmap with ARGB_8888 to avoid Config.HARDWARE issue
        return originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
    }



    private void runModel(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        inputBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[224 * 224];
        resized.getPixels(pixels, 0, 224, 0, 0, 224, 224);

        for (int pixel : pixels) {
            float r = ((pixel >> 16) & 0xFF) / 255.f;
            float g = ((pixel >> 8) & 0xFF) / 255.f;
            float b = (pixel & 0xFF) / 255.f;

            inputBuffer.putFloat(r);
            inputBuffer.putFloat(g);
            inputBuffer.putFloat(b);
        }
        inputBuffer.rewind();

        float[][] output = new float[1][classLabels.length];
        tflite.run(inputBuffer, output);

        int maxIndex = 0;
        float maxProb = output[0][0];
        for (int i = 1; i < output[0].length; i++) {
            if (output[0][i] > maxProb) {
                maxProb = output[0][i];
                maxIndex = i;
            }
        }

        String resultStr = String.format("Prediction: %s\nConfidence: %.2f%%", classLabels[maxIndex], maxProb * 100);
        resultTextView.setText(resultStr);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
    }
}
