**Traversing the DOM**

Now it's time to dig into some HTML and select some DOM elements.

You'll do this all the time in your code as a front end developer so it's good to know how to use the selectors and methods.

In this exercise you're going to dive into the DOM of an existing website and select specific elements. The webpage will look like this...

![DOM traversal screenshot](./images/DOM_selection_screenshot.jpg)

It's nothing special, but it will do the job when it comes to letting you test out your DOM navigation skills.

At the bottom of the page will be a series of questions asking you to target specific elements on the page. You'll type your query in the input and once you submit you can see whether your target matches the expected target.

There may be multiple ways you can get to each element so the most important thing is that the elements match. The queries can be very different but as long as the end result is the same, that's all that matters.

Having said that, there are some rules...

**RULES:**
- Each query can only be one line.
  - You can chain methods as you need to. In fact one question even asks you to chain a method.
- Each query MUST start with the document root.
- Each query MUST use .querySelector() or .querySelectorAll()
- No illegal characters
  -inputs will be sanitized before checking

**Getting started:**

Go to this GitHub repository.

Once there, you'll need to clone these files into your local machine.

Once you have the repo cloned locally, you can right click on the index.html and open it in your browser.

You'll definitely need to open your developer tools to see the HTML structure and complete the assignment.

Then simply enter your query for each question and see if yours matches.