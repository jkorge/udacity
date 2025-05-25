# Cypress

## **Summary**

In this exercise, you will write end-to-end (E2E) tests using Cypress to test the gallery page of your application.

---

### **Resources**

- [Cypress Documentation](https://docs.cypress.io/app/get-started/why-cypress#End-to-end-Testing)

---

## **Instructions**

We will be testing the gallery page of our application.

### **Prepare the Application**

1. Open **two terminal windows**:
   - In the first terminal, run:
     ```bash
     npm run start
     npm i
     ```
     This starts the server at [http://localhost:1234/](http://localhost:1234/).
   - Use the second terminal for all other commands.

---

### **1. Install Cypress**

- In the second terminal, run:
  ```bash
  npm install cypress --save-dev
  ```

---

### **2. Run Cypress Test Runner**

- In the terminal, run:
  ```bash
  npx cypress open
  ```
- This opens the Cypress Test Runner window.
- Click **Continue**.
- Select **Start E2E Testing in Electron** (the default testing browser).

---

### **3. Create a Spec**

- In the Test Runner, select **Create New Spec**.
- Name the spec `priceForm.cy.js` and click **Create**.
- Click **Okay, Run the Spec** to run the prebuilt example spec.

---

### **4. Build a Test**

1. Open the `priceForm.cy.js` file in VS Code (located under `cypress/e2e/`).
2. Update the `describe` and `it` blocks to reflect the test suite and test case:
   ```javascript
   describe("Gallery Page Functionality", () => {
     it("Gallery loads with images", () => {
       cy.visit("http://localhost:1234/");
     });
   });
   ```
3. Update `cy.visit` with your server's URL: `http://localhost:1234/`.
4. Verify the application appears in the Cypress Test Runner.

---

### **5. Navigate to the Gallery Page**

1. Use `cy.get()` to select the gallery element with the attribute `[data-cy="galleryPage"]`:
   ```javascript
   cy.get('[data-cy="galleryPage"]').click();
   ```
2. Verify in the Test Runner that the application has navigated to the gallery page.

Full code so far:
```javascript
describe("Gallery Page Functionality", () => {
  it("Gallery loads with images", () => {
    cy.visit("http://localhost:1234/");
    cy.get('[data-cy="galleryPage"]').click();
  });
});
```

---

### **6. Verify Images Are Visible**

1. Use `cy.get()` to select the gallery container with the attribute `[data-cy="container"]`.
2. Chain `.find("img")` to locate all image elements within the container.
3. Use `.each()` to iterate over each image:
   - Pass an anonymous callback with a parameter `img`.
   - Use `cy.wrap(img)` to provide Cypress functionality to the element.
   - Chain `.should("be.visible")` to assert that each image is visible.

Full code:
```javascript
describe("Gallery Page Functionality", () => {
  it("Gallery loads with images", () => {
    cy.visit("http://localhost:1234/");
    cy.get('[data-cy="galleryPage"]').click();
    cy.get('[data-cy="container"]')
      .find("img")
      .each((img) => {
        cy.wrap(img).should("be.visible");
      });
  });
});
```

**Note**: 
- The `.wrap()` function is necessary because `.each()` iterates over raw DOM elements, which lack Cypress functionality. Wrapping an element re-enables Cypress commands like `.should()`.

---

### **7. Run the Test**

1. Return to the Cypress Test Runner.
2. Run the updated test and verify that all assertions pass.