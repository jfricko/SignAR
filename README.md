# SignAR

**SignAR: Real-time recognition and AR visualization of Filipino Sign Language**

A browser-based prototype for real-time Filipino Sign Language gesture recognition using the device camera, MediaPipe hand landmarks, adaptive examples, sentence construction, and browser voice output.

## GitHub Pages

This project is prepared as a static website and can be deployed directly with GitHub Pages.

1. Create a GitHub repository named `SignAR`.
2. Upload the **contents of this folder** to the repository root.
3. Open **Settings → Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Select the `main` branch and `/(root)` folder, then save.
6. Open the GitHub Pages URL shown by GitHub.

Keep `index.html` at the repository root. Do not place the project inside an extra nested folder.

## Main features

- Filipino Sign Language-focused recognition vocabulary.
- Live camera hand tracking with MediaPipe Hands.
- Front/back camera switching on supported phones.
- AR-style recognition feedback over the live camera.
- Automatic confirmed-gesture sentence building.
- Cleaner sentence output with basic Filipino/English translation handling.
- Filipino or English browser speech output.
- Optional automatic speech when a sign is confirmed.
- No recognition-history interface.
- No English Sign Language alphabet mode.
- No separate sign-library page in this GitHub-ready build.
- Responsive mobile-friendly layout.
- PWA manifest and service worker.

## Important note

The included recognizer is a prototype with a limited vocabulary and heuristic/adaptive landmark logic. It should not be presented as a complete Filipino Sign Language translator without validation using authentic FSL data and testing with qualified FSL signers or experts.

## Camera requirements

Camera access requires browser permission and a secure context. GitHub Pages provides HTTPS, so it is suitable for phone camera testing after deployment.

## Project structure

```text
SignAR/
├── index.html
├── script.js
├── style.css
├── manifest.webmanifest
├── sw.js
├── .nojekyll
├── README.md
└── icons/
    └── icon.svg
```

## External dependency

The browser build loads MediaPipe Hands from jsDelivr. An internet connection is required for the external library unless it is later bundled locally.
