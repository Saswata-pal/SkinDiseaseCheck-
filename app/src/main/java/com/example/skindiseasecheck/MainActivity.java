package com.example.skindiseasecheck;

import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.text.HtmlCompat;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    // --- Tunable thresholds ---
    private static final float SKIN_PIXEL_RATIO_THRESHOLD = 0.12f; // min fraction of skin-toned pixels
    private static final float CONFIDENCE_THRESHOLD = 0.55f;       // min model confidence to trust a label

    private ImageView imageView;
    private TextView placeholderText;
    private ProgressBar progressBar;
    private MaterialButton btnSelectImage;
    private MaterialCardView resultCard;
    private TextView predictionLabel;
    private TextView predictionConfidence;

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

        showAboutDialog();

        imageView = findViewById(R.id.imageView);
        placeholderText = findViewById(R.id.placeholderText);
        progressBar = findViewById(R.id.progressBar);
        btnSelectImage = findViewById(R.id.selectImageBtn);
        resultCard = findViewById(R.id.resultCard);
        predictionLabel = findViewById(R.id.predictionLabel);
        predictionConfidence = findViewById(R.id.predictionConfidence);

        try {
            tflite = new Interpreter(FileUtil.loadMappedFile(this, "skin_disease_model.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
            showResult("Error loading model", e.getMessage(), null);
        }

        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri == null) {
                        // User cancelled the picker — restore the placeholder
                        // only if there's no image currently shown.
                        if (imageView.getDrawable() == null) {
                            placeholderText.setVisibility(View.VISIBLE);
                        }
                        return;
                    }
                    try {
                        Bitmap bitmap = getBitmapFromUri(uri);
                        imageView.setImageBitmap(bitmap);
                        placeholderText.setVisibility(View.GONE);
                        analyzeImage(bitmap);
                    } catch (IOException e) {
                        e.printStackTrace();
                        showResult("Failed to load image", "Please try a different photo.", null);
                    }
                });

        btnSelectImage.setOnClickListener(v -> imagePickerLauncher.launch("image/*"));
    }

    private void showAboutDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
        builder.setTitle(getString(R.string.about_title));
        View dialogView = getLayoutInflater().inflate(R.layout.about_dialog, null);
        TextView aboutTextView = dialogView.findViewById(R.id.aboutTextView);
        aboutTextView.setText(HtmlCompat.fromHtml(getString(R.string.about_description), HtmlCompat.FROM_HTML_MODE_LEGACY));
        builder.setView(dialogView);
        builder.setPositiveButton("OK", null);
        builder.show();
    }

    private Bitmap getBitmapFromUri(Uri uri) throws IOException {
        Bitmap originalBitmap;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            ImageDecoder.Source source = ImageDecoder.createSource(getContentResolver(), uri);
            originalBitmap = ImageDecoder.decodeBitmap(source);
        } else {
            originalBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
        }
        return originalBitmap.copy(Bitmap.Config.ARGB_8888, true);
    }

    private void analyzeImage(Bitmap bitmap) {
        progressBar.setVisibility(View.VISIBLE);
        resultCard.setVisibility(View.GONE);
        btnSelectImage.setEnabled(false);

        // Run on the UI thread is fine here since the heuristic + tflite inference
        // on a single small image are both fast, but you could move this to a
        // background thread/executor if you notice any jank on lower-end devices.
        imageView.post(() -> {
            if (!isLikelySkin(bitmap)) {
                showResult(
                        "No skin detected",
                        "This doesn't look like a photo of skin. Please upload a clear, close-up photo of the affected area.",
                        null
                );
            } else {
                runModel(bitmap);
            }
            progressBar.setVisibility(View.GONE);
            btnSelectImage.setEnabled(true);
        });
    }

    /**
     * Quick heuristic gatekeeper: samples pixels from the image and checks what
     * fraction fall inside a typical human-skin range in YCbCr color space.
     * This won't be perfect, but it reliably filters out things like screenshots,
     * documents, landscapes, food, pets, etc. before they hit the classifier.
     */
    private boolean isLikelySkin(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        if (width == 0 || height == 0) return false;

        // Downsample for speed: cap sampling density regardless of image resolution
        int step = Math.max(1, Math.min(width, height) / 100);

        int totalSamples = 0;
        int skinPixels = 0;

        for (int y = 0; y < height; y += step) {
            for (int x = 0; x < width; x += step) {
                int pixel = bitmap.getPixel(x, y);
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;

                totalSamples++;
                if (isSkinColor(r, g, b)) {
                    skinPixels++;
                }
            }
        }

        if (totalSamples == 0) return false;

        float ratio = (float) skinPixels / totalSamples;
        return ratio >= SKIN_PIXEL_RATIO_THRESHOLD;
    }

    private boolean isSkinColor(int r, int g, int b) {
        // RGB -> YCbCr
        double cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
        double cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;

        // Standard empirical skin range in YCbCr (covers a broad range of skin tones)
        return cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173;
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

        if (maxProb < CONFIDENCE_THRESHOLD) {
            showResult(
                    "Uncertain result",
                    String.format("The model isn't confident enough to give a reliable prediction (%.0f%%). Try a clearer, well-lit, close-up photo.", maxProb * 100),
                    null
            );
        } else {
            showResult(
                    classLabels[maxIndex],
                    String.format("Confidence: %.1f%%", maxProb * 100),
                    maxProb
            );
        }
    }

    /**
     * Updates and shows the result card. Also color-codes the card's stroke:
     * teal = confident prediction, amber = uncertain, gray = no skin / error.
     */
    private void showResult(String title, String subtitle, Float confidence) {
        predictionLabel.setText(title);
        predictionConfidence.setText(subtitle == null ? "" : subtitle);

        int colorRes;
        if (confidence != null) {
            colorRes = R.color.result_positive;
        } else if ("Uncertain result".equals(title)) {
            colorRes = R.color.result_warning;
        } else {
            colorRes = R.color.result_neutral;
        }

        int color = ContextCompat.getColor(this, colorRes);
        resultCard.setStrokeColor(color);
        resultCard.setStrokeWidth(4);

        resultCard.setVisibility(View.VISIBLE);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tflite != null) {
            tflite.close();
        }
    }
}