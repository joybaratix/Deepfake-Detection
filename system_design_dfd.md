# System Design & Development DFD (Terminal Notepad Style)

Here is your layer-wise system design diagram in a classic "computer notepad" and "terminal drawing" ASCII aesthetic, featuring a pure white background and deep dark colors for each layer.

### Image View (9:16 Ratio Snapshot)

This image is beautifully highlighted with dark deep text colors and formatted in an exact 9:16 aspect ratio, perfectly suited for vertical displays.

You can right-click the image below to copy or save it!

![Dark Deep Color ASCII DFD (9:16 Ratio)](/C:/Users/joyba/.gemini/antigravity-ide/brain/fcc76337-85e0-4f09-b997-c9f6214a4ada/dfd_notepad_ascii_9_16.png)

---

### Plain Text Version

If you need to copy and paste the dotted-line drawing directly into your code or a `.txt` file, you can copy the expanded text below:

```text
=============================================================================
                        SYSTEM DESIGN: LAYER-WISE DFD
                        (Deepfake Detection Using CNN)
=============================================================================



            [ PRESENTATION LAYER ]
            .-------------------------------------------------------.
            |                                                       |
            |  User Interface (Web App)                             |
            |  - Media Upload                                       |
            |  - Result Display                                     |
            |                                                       |
            '-------------------------------------------------------'
                       |                                 ^
                       |                                 |
          Raw Media    |                                 |  Final Result
        (Video/Image)  |                                 |  (Real/Fake)
                       |                                 |
                       v                                 |
            [ APPLICATION LAYER ]                        |
            .-------------------------------------------------------.
            |                                                       |
            |  Preprocessing Module                                 |
            |  - Video Frame Extraction                             |
            |  - Face Detection & Cropping                          |
            |  - Normalization                                      |
            |                                                       |
            '-------------------------------------------------------'
                       |                                 ^
                       |                                 |
        Preprocessed   |                                 |  Prediction
            Faces      |                                 |    Score
                       |                                 |
                       v                                 |
            [ MODEL LAYER ]                              |
            .-------------------------------------------------------.
            |                                                       |
            |  CNN Architecture                                     |
            |  - Feature Extractor (e.g., MesoNet)                  |
            |  - Classification Head                                |
            |                                                       |
            '-------------------------------------------------------'
                       |                                 ^
                       |                                 | Load Weights
                       |                                 |
                       v                                 |
            [ DATA LAYER ]                               |
            .-------------------------------------------------------.
            |                                                       |
            |  Storage                                              |
            |  - Uploaded Media Cache                               |
            |  - Pretrained CNN Weights                             |
            |                                                       |
            '-------------------------------------------------------'
```
