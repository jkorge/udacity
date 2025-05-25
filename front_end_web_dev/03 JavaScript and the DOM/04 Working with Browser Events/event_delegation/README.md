**Event Delegation Exercise**
---
Now it's your turn to create your own event listener using event delegation.

This shouldn't be too hard. You only need one listener and one handler. But you do need to slow down and think about the best way to make it work.

You're going to be given some HTML of product cards for an online marketplace. Your goal, for now is to create an event listener that will open an alert that says "You selected the insert item here. Good choice!"

In case it's not clear, you'll need to insert the title of the item where I said "insert item here."

It should work like this example...

<iframe width="800" height="800" src="https://www.youtube.com/embed/sSqQczBOi3I" title="" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For this exercise you can only have..
- 1 event listener
- 1 event handler
  - remember to think about other devices and use an event that works with multiple devices well
  - use best practices when naming your handler

Making this work with only 1 listener is going to force you to think about how the event processor works. Remember that just because you click on what looks like the card doesn't mean that's where the event is actually fired. If the event is recorded on a child of the product card, how do you make that work?

You'll have to move up and down the DOM using the methods we've already learned about in order to get the alert to display properly. (HINT: Check lesson 2 on traversing the DOM)

**Requirements:**
---
For this exercise you have to have only one event listener.

You can only have one named (not anonymous) event handler.

Do not modify the provided HTML or CSS in any way.

The only file you should modify is the scripts.js file.

**Getting started:**
---
In order to work on this project you'll need to go to this GitHub repository.

lesson-4-event-delegation/exercises/starter

This folder holds all your starter files you'll need.

The ONLY file you need to edit is the scripts.js file in the js folder. Don't make any changes to HTML or CSS.

Remember this is an exercise to help you implement event delegation effectively, so focus on that.