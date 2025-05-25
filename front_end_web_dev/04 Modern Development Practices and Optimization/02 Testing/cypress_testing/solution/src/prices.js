import { clearMain, calculatePackagePrice} from "./helperFunctions.js"

export const renderPricePage = () => {
  clearMain()

  const photoData = [
    { type: "portrait", size: "4\" x 6\"", basePricePerCopy: 0.50 },
    { type: "portrait", size: "8\" x 10\"", basePricePerCopy: 1.50 },
    { type: "portrait", size: "11\" x 14\"", basePricePerCopy: 3.00 },
    { type: "landscape", size: "5\" x 7\"", basePricePerCopy: 0.75 },
    { type: "landscape", size: "16\" x 20\"", basePricePerCopy: 10.00 },
    { type: "panorama", size: "12\" x 36\"", basePricePerCopy: 5.00 },
    { type: "square", size: "6\" x 6\"", basePricePerCopy: 1.00 },
    { type: "passport", size: "2\" x 2\"", basePricePerCopy: 0.25 },
    { type: "poster", size: "24\" x 36\"", basePricePerCopy: 15.00 },
    { type: "wallet", size: "2.5\" x 3.5\"", basePricePerCopy: 0.20 }
  ];
  const form = renderPriceEstimateForm(photoData);
  form.addEventListener('submit', (e) => {
    const photoArray = handleCalculatePrice(e)
    console.log(photoArray)
    const total = calculatePackagePrice(photoArray)
    console.log(total)
    const h2 = document.createElement('h2')
    h2.textContent = `Photo Package Price Estimate $${total.toFixed(2)}`
    h2.dataset.cy = "priceEstimate"
    console.log(h2)
    document.querySelector('main').append(h2)

})
  document.querySelector('main').append(form)

}

const renderPriceEstimateForm = (photoData) => {
  // Create the form element
  const form = document.createElement("form");
  form.id = "form";
  form.dataset.cy = 'packagePriceForm'

  // Add a heading to the form
  const heading = document.createElement("h2");
  heading.textContent = "Price Estimate Form";
  form.appendChild(heading);

  // Loop through the photo data to create form fields
  photoData.forEach((photo, index) => {
    // Create a container for each photo type
    const container = document.createElement("div");
    container.className = "photo-option";

    // Create a label for the photo type and size
    const label = document.createElement("label");
    const uniqueId = `photo-${index}`; // Generate a unique id
    label.htmlFor = uniqueId;
    label.textContent = `${photo.type} (${photo.size}) - $${photo.basePricePerCopy.toFixed(2)} per copy`;

    // Create a number input for the number of copies
    const input = document.createElement("input");
    input.type = "number";
    input.name = `copies-${photo.type}-${photo.size}`;
    input.id = uniqueId;
    input.min = 0; // Prevent negative numbers
    input.placeholder = "Enter number of copies";
    

    // Append the label and input to the container
    container.appendChild(label);
    container.appendChild(input);

    // Append the container to the form
    form.appendChild(container);
  });

  // Add a submit button
  const submitButton = document.createElement("button");
  submitButton.type = "submit";
  submitButton.textContent = "Calculate Price";
  form.appendChild(submitButton);

  // Return the form element
  return form;
};

const handleCalculatePrice = (e) => {
  //Prevents forms default behavor to refresh the page
  e.preventDefault()
  const photoArray = [];
  //Selects all of the inputs from the form
  const inputs = e.target.querySelectorAll("input[type='number']")
  //Iterates through the form
  inputs.forEach(input => {
    //Gets the lable for the form
    const label = form.querySelector(`label[for="${input.id}"]`);
    //Checks the input for a value 
    //If found keys and values will be estracted form the lable and input
  
    if (label && input.value > 0) {
      const labelText = label.textContent;
      const [type, details] = labelText.split(" (");
      const [size, priceText] = details.split(") - $");
      const basePricePerCopy = parseFloat(priceText.split(" ")[0]);
      //photo object created and formated 
      const photoOrder = {
        type: type.trim(),
        size: size.trim(),
        copies: parseInt(input.value, 10), // Convert value to an integer
        basePricePerCopy,
      };
      //added to photo array
      photoArray.push(photoOrder)
    }
  })
  return photoArray
}