// Do not modify this file.
// This function is inspired by the popular "FizzBuzz" algorithm question.
// - If the function is passed a number divisible by 3, it returns "Fizz."
// - If the number is divisible by 5, it returns "Buzz."
// - If the number is divisible by both 3 and 5, it returns "FizzBuzz."
// - If the number is not divisible by 3 or 5, it returns the number itself.
// - If the input is not a number, the function returns false.
export const fizzBuzz = (n) => {
  if (typeof n !== "number" || isNaN(n)) {
    return false;
  }
  if (n % 3 === 0 && n % 5 === 0) {
    return "FizzBuzz";
  } else if (n % 3 === 0) {
    return "Fizz";
  } else if (n % 5 === 0) {
    return "Buzz";
  } else {
    return n;
  }
};
