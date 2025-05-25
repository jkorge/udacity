//Don't Touch anything in this file
// This function takes a password as an argument
// and returns true if the password passes 3 levels of validation:
// Password must be at least 12 characters long.
// Password must contain at least one uppercase letter.
// Password must contain at least one special character.

export const passwordValidator = (password) => {
  const minLength = 12;

  // Regular expressions to check for uppercase letters and special characters
  const uppercasePattern = /[A-Z]/;
  const specialCharPattern = /[!@#$%^&*(),.?":{}|<>]/;

  // Check if the password meets the minimum length requirement
  if (password.length < minLength) return false;

  // .test() checks if the regex pattern matches the password string
  if (!uppercasePattern.test(password)) return false;
  if (!specialCharPattern.test(password)) return false;

  // Return true if all validations pass
  return true;
};
