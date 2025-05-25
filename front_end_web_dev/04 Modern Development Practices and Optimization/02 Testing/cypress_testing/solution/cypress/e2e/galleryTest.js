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
