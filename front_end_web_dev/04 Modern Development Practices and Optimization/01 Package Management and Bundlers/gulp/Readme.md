# Gulp

## **Summary**

In this exercise, you will use Gulp to automate testing with Mocha.

---

## **Resources**

- [Gulp Documentation](https://gulpjs.com/)

---

## **Instructions**

### **1. Install Gulp**

1. Open the terminal and follow these steps:
   - Install Gulp CLI globally:
     ```
     npm install -g gulp-cli
     ```
   - Install Gulp and Gulp Shell locally:
     ```
     npm install gulp gulp-shell --save-dev
     ```

### **2. Create a Gulp Task**

1. Create a `gulpfile.js` in your project root.
2. Add the following:
   - Import Gulp and Gulp Shell:
     ```
     const gulp = require("gulp");
     const shell = require("gulp-shell");
     ```
   - Define a Gulp task for running tests:
     ```
     gulp.task("unit test", shell.task("mocha"));
     ```

### **3. Run Gulp**

1. In the terminal, run the following command:
   ```
   gulp "unit test"
   ```
