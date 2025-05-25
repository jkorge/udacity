# Parcel

## **Summary**

In this exercise, you will configure Parcel to bundle your application.

---

### **Resources**

- [Parcel Documentation](https://parceljs.org/getting-started/library/)

---

## **Instructions**

### **1. Install Parcel**

- Open the terminal and run:
  ```bash
  npm install --save-dev parcel
  ```

### **2. Configure Parcel**

- Add the following keys to your `package.json`:
  ```
  "source": "src/index.js",
  "main": "dist/main.js"
  ```
  **Note**:
  - `source` specifies the entry point for Parcel (your main JavaScript file).
  - `main` defines the output file where the bundled code will be generated.

### **3. Create Build Script**

- Add the following to the `scripts` section of your `package.json`:
  ```
  "build": "parcel build"
  ```

### **4. Run the Build**

- In the terminal, run:
  ```
  npm run build
  ```

### **5. Update Your HTML**

- Update the `<script>` tag in your HTML file to reference the bundled file:
  ```
  <script src="dist/main.js"></script>
  ```

### **6. Verify the Application**

- Open the HTML file in a browser and ensure your JavaScript functions are working correctly.
