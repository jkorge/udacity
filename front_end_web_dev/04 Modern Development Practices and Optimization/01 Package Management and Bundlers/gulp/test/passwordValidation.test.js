//Don't Touch anything in this file 
// test/passwordValidator.test.js
import { expect } from "chai";
import { passwordValidator } from "../src/passwordValidation.js";

// These tests validate the functionality of the passwordValidator function.
// Mocha is used as the testing framework, and Chai provides assertion capabilities.
// We will cover these tools in a later lesson
describe("Password Validator", () => {
  it("should return true for a valid password", () => {
    const validPassword = "StrongPass123!";
    expect(passwordValidator(validPassword)).to.be.true;
  });

  it("should return false for a password shorter than 12 characters", () => {
    const shortPassword = "Short1!";
    expect(passwordValidator(shortPassword)).to.be.false;
  });

  it("should return false for a password without an uppercase letter", () => {
    const noUppercase = "nouppercase123!";
    expect(passwordValidator(noUppercase)).to.be.false;
  });

  it("should return false for a password without a special character", () => {
    const noSpecialChar = "NoSpecialChar123";
    expect(passwordValidator(noSpecialChar)).to.be.false;
  });

  it("should return false for an empty password", () => {
    const emptyPassword = "";
    expect(passwordValidator(emptyPassword)).to.be.false;
  });
});
