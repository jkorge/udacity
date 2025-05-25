**Adding Content to the DOM**
---
Let's visit our friend Whiskers again.

This time when we open his page, it looks like this...

![No page for Whiskers](./images/no-whiskers.jpg)

That's not good at all. 

Your challenge this time is to use JavaScript to create and add all the elements you need in order to make the page look like this...

![Finished page for Whiskers](./images/styling_result.jpg)

That's right. We're going to build on what you did in the earlier exercise and fix Whiskers' page.

The styling should be identical so your main goal will be to add code that will create and add elements efficiently.

Take your time with this exercise!!

There are a lot of moving pieces and you need to make sure you add all the elements correctly in order for the page to look like we expect. Go one step at a time and make sure you keep track of what level you're creating and adding content to.

**Getting started:**
---
As usual, your exercise will be in this GitHub repository.

Clone this repo to your local system and then you'll have access to the code.

**DO NOT EDIT THE HTML OR CSS.**

index.html and styles.css should not be modified in any way.

In the JS folder, you'll find a file called scripts.js. In that file, you'll see an empty function called rebuildOtter().
This is where all your code should go.

When you open up the index.html file in your local system you can keep track of how you're doing.

**Tips:**
---
Did I mention to take your time?

If this is your first time really building like this it can be easy to get lost in the weeds. If you need to, write out each element you need to create as well as its location in the DOM.

Check your developer tools frequently as you go. Not only will you see any possible errors show up, you can look at the elements and see how the page is coming along. Use the earlier exercise as a guide to how your finished HTML should look.

Use a document fragment to add your final changes. You are going to make a lot of new elements and if you don't use a fragment than every change you make will cause a layout and repaint of the page. Just creating your elements doesn't affect the DOM, so create your elements in their finished form and then add them to your fragment before adding them to the DOM. That will end up with only one reflow and one repaint, which is pretty good considering the number of changes you're making.

Don't forget to add styling to your elements. The styling should be identical to the styling in your previous exercise for 'Modifying Element Styling.' Look back to that exercise and make sure your classes and ids match what's expected.