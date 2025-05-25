// test/passwordValidator.test.js
import { expect } from "chai";
import { fizzBuzz } from "../src/fizzBuzz.js";

describe("FizzBuzz Function", () => {
  it("Should return 'FizzBuzz' if the input is divisible by 3 and 5", () => {
    expect(fizzBuzz(15)).to.equal("FizzBuzz");
    expect(fizzBuzz(30)).to.equal("FizzBuzz");
  });

  it("should return 'Fizz' if the input is only divisible by 3", () => {
    expect(fizzBuzz(3)).to.equal("Fizz");
    expect(fizzBuzz(9)).to.equal("Fizz");
  });

  it("should return 'Buzz' if the input is only divisible by 5", () => {
    expect(fizzBuzz(5)).to.equal("Buzz");
    expect(fizzBuzz(10)).to.equal("Buzz");
  });

  it("should return false when the argument is not a number", () => {
    expect(fizzBuzz("hello")).to.be.false;
    expect(fizzBuzz("123")).to.be.false;
    expect(fizzBuzz(true)).to.be.false;
    expect(fizzBuzz([])).to.be.false;
  });
});
