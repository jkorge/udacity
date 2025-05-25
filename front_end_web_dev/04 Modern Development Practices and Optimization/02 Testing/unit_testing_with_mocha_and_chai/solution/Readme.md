# Unit Tests with Mocha

## **Summary**

In this exercise, you will create unit tests using Mocha and Chai.

---

### **Resources**

- [Mocha Documentation](https://mochajs.org/)
- [Chai Documentation](https://www.chaijs.com/)

---

## **Instructions**

`src/fizzBuzz.js` includes a function we will be testing with Mocha.

---

### **1. Install Mocha and Chai**

- Open the terminal and enter the following command to install Mocha and Chai as development dependencies:
  ```
  npm install mocha chai --save-dev
  ```

---

### **2. Create a Test File**

- Create a new file in the `testing` folder called `fizzBuzz.test.js`.

---

### **3. Build the Testing Suite**

- Import Chai and the function to be tested:

  ```
  const { expect } = require("chai");
  const fizzBuzz = require("../src/fizzBuzz.js");
  ```

  **Note**: Mocha is a test runner and does not need to be imported.

- Create a `describe` block for your test suite:
  ```
  describe("FizzBuzz Function", () => {
    // Test cases will go here
  });
  ```

---

### **4. Create a Test Case**

- Inside the `describe` block, create an `it` block for a specific test case:
  ```
  describe("FizzBuzz Function", () => {
    it("Should return 'FizzBuzz' if the input is divisible by 3 and 5", () => {
      // Assertions will go here
    });
  });
  ```

---

### **5. Add an Assertion**

- Call the `fizzBuzz` function, passing `15` as an argument, and save the result to a variable:

  ```
  const result = fizzBuzz(15);
  ```

- Use Chai's `expect` to make an assertion:

  ```
  expect(result).to.equal("FizzBuzz");
  ```

- Full test case:
  ```
  describe("FizzBuzz Function", () => {
    it("Should return 'FizzBuzz' if the input is divisible by 3 and 5", () => {
      const result = fizzBuzz(15);
      expect(result).to.equal("FizzBuzz");
    });
  });
  ```

---

### **6. Run Your Tests**

- Add a test script to your `package.json`:

  ```
  "scripts": {
    "test": "mocha"
  }
  ```

- Open the terminal and run:

  ```
  npm run test
  ```

- You should see output like this:

  ```
   FizzBuzz Function
      ✔ Should return 'FizzBuzz' if the input is divisible by 3 and 5

    1 passing (3ms)
  ```

---

### **7. Add More Test Cases**

- Repeat steps **4–6** for the following scenarios:

  1. Numbers divisible by 3 should return `"Fizz"`.
  2. Numbers divisible by 5 should return `"Buzz"`.
  3. Non-number inputs should return `false`.

- Use the [Chai Documentation](https://www.chaijs.com/api/bdd/) to explore other assertions and methods for validation.
