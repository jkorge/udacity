import { renderHomePage } from "./home.js";
import { renderGalleryPage } from "./gallery.js";
import { renderPricePage } from "./prices.js";

// Navigation logic
const addListenerToNavigation = () => {
  addNavListener("homePage", renderHomePage);
  addNavListener("galleryPage", renderGalleryPage);
  addNavListener("pricePage", renderPricePage);
};

// Helper function to add event listeners to navigation items
const addNavListener = (elementId, callback) => {
  const element = document.getElementById(elementId);
  if (element) {
    element.addEventListener("click", callback);
  }
};

export const renderNavigation = () => {
  addListenerToNavigation();
};
