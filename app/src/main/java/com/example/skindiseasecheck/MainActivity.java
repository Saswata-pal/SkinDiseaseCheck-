package com.example.skindiseasecheck;

import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
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
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    // --- Tunable thresholds ---
    private static final float SKIN_PIXEL_RATIO_THRESHOLD = 0.08f; // min fraction of skin-toned pixel
    private static final float CONFIDENCE_THRESHOLD = 0.35f;       // min model confidence to trust a label
    private static final int TEXT_BLOCK_COUNT_THRESHOLD = 3;       // min distinct text blocks to flag as screenshot/doc
    private static final int TEXT_CHAR_COUNT_THRESHOLD = 20;       // min total recognized characters to flag as screenshot/doc

    private static final String TAG = "SkinModel";

    private ImageView imageView;
    private TextView placeholderText;
    private ProgressBar progressBar;
    private MaterialButton btnSelectImage;
    private MaterialCardView resultCard;
    private TextView predictionLabel;
    private TextView predictionConfidence;

    private Interpreter tflite;
    private FaceDetector faceDetector;
    private TextRecognizer textRecognizer;

    private final String[] classLabels = {
            "Eczema", "Warts/Viral Infections", "Melanoma", "Atopic Dermatitis",
            "BCC", "Melanocytic Nevi", "Benign Lesions", "Psoriasis/Lichen Planus",
            "Seborrheic Keratoses", "Fungal Infections"
    };

    private ActivityResultLauncher<String> imagePickerLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Backup enforcement in case the OS still tries to force-dark this activity
        // despite the manifest flag (some OEM skins are stubborn about this).
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            getWindow().getDecorView().setForceDarkAllowed(false);
        }

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

        FaceDetectorOptions faceOptions = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .build();
        faceDetector = FaceDetection.getClient(faceOptions);

        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);

        imagePickerLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri == null) {
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

    /**
     * Pipeline: face check -> text/screenshot check -> skin-color check -> model.
     * Each stage runs only if the previous one passes.
     */
    private void analyzeImage(Bitmap bitmap) {
        progressBar.setVisibility(View.VISIBLE);
        resultCard.setVisibility(View.GONE);
        btnSelectImage.setEnabled(false);

        InputImage inputImage = InputImage.fromBitmap(bitmap, 0);

        faceDetector.process(inputImage)
                .addOnSuccessListener(faces -> {
                    if (faces != null && !faces.isEmpty()) {
                        Log.d(TAG, "Face(s) detected: " + faces.size() + " — rejecting as portrait/face photo.");
                        showResult(
                                "This looks like a face or portrait",
                                "Please upload a close-up photo of just the affected skin area, without a face in frame.",
                                null
                        );
                        finishAnalysis();
                    } else {
                        checkForText(bitmap);
                    }
                })
                .addOnFailureListener(e -> {
                    Log.w(TAG, "Face detection failed, proceeding without it: " + e.getMessage());
                    checkForText(bitmap);
                });
    }

    /**
     * Runs on-device OCR. Screenshots, documents, and UI mockups almost always
     * contain a good amount of rendered text; real skin photos essentially never
     * do. If enough text is found, we reject the image before it reaches the
     * skin-color heuristic or the classifier.
     */
    private void checkForText(Bitmap bitmap) {
        InputImage inputImage = InputImage.fromBitmap(bitmap, 0);

        textRecognizer.process(inputImage)
                .addOnSuccessListener(visionText -> {
                    int blockCount = visionText.getTextBlocks().size();
                    int charCount = visionText.getText().replaceAll("\\s", "").length();

                    Log.d(TAG, String.format("Text blocks: %d, char count: %d", blockCount, charCount));

                    boolean looksLikeTextHeavyImage =
                            blockCount >= TEXT_BLOCK_COUNT_THRESHOLD || charCount >= TEXT_CHAR_COUNT_THRESHOLD;

                    if (looksLikeTextHeavyImage) {
                        showResult(
                                "This looks like a screenshot or document",
                                "Please upload an actual photo of the affected skin area, not a screenshot or text-based image.",
                                null
                        );
                        finishAnalysis();
                    } else {
                        continueWithSkinCheck(bitmap);
                    }
                })
                .addOnFailureListener(e -> {
                    Log.w(TAG, "Text recognition failed, proceeding without it: " + e.getMessage());
                    continueWithSkinCheck(bitmap);
                });
    }

    private void continueWithSkinCheck(Bitmap bitmap) {
        imageView.post(() -> {
            boolean looksLikeSkin = isLikelySkin(bitmap);
            Log.d(TAG, "isLikelySkin() result: " + looksLikeSkin);

            if (!looksLikeSkin) {
                showResult(
                        "No skin detected",
                        "This doesn't look like a photo of skin. Please upload a clear, close-up photo of the affected area.",
                        null
                );
            } else {
                runModel(bitmap);
            }
            finishAnalysis();
        });
    }

    private void finishAnalysis() {
        progressBar.setVisibility(View.GONE);
        btnSelectImage.setEnabled(true);
    }

    private boolean isLikelySkin(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        if (width == 0 || height == 0) return false;

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
        Log.d(TAG, String.format("Skin pixel ratio: %.3f (threshold: %.3f)", ratio, SKIN_PIXEL_RATIO_THRESHOLD));

        return ratio >= SKIN_PIXEL_RATIO_THRESHOLD;
    }

    private boolean isSkinColor(int r, int g, int b) {
        double cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
        double cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;

        boolean normalSkin = cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173;
        boolean inflamedOrScaly = cb >= 70 && cb <= 135 && cr >= 128 && cr <= 190;

        return normalSkin || inflamedOrScaly;
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

        StringBuilder debug = new StringBuilder("Model output:\n");
        for (int i = 0; i < classLabels.length; i++) {
            debug.append(String.format("  %s: %.1f%%\n", classLabels[i], output[0][i] * 100));
        }
        Log.d(TAG, debug.toString());

        int maxIndex = 0;
        float maxProb = output[0][0];
        for (int i = 1; i < output[0].length; i++) {
            if (output[0][i] > maxProb) {
                maxProb = output[0][i];
                maxIndex = i;
            }
        }

        Log.d(TAG, String.format("Top prediction: %s (%.1f%%)", classLabels[maxIndex], maxProb * 100));

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
        if (faceDetector != null) {
            faceDetector.close();
        }
        if (textRecognizer != null) {
            textRecognizer.close();
        }
    }
}