describe('Navigation', () => {

    before(() => {
        cy.visit('http://localhost:1234/');
    });

    it('Should load pages selected from the nav bar', () => {

        // Nav bar should be visible
        cy.get('nav').should('be.visible');

        // Try each nav link
        cy.get('nav>li')
            .each(($elem) => {

                // Get the ID of the `li`
                const pageID = $elem.attr('id');

                // Click the `li`
                cy.wrap($elem)
                    .click()
                    .then(() => {
                        // Check that the relevant DOM object is visible for each page
                        // NB: the card page's main `div` has a different naming scheme than the other pages (hence the ternary expression)
                        cy.get(`.${(pageID == 'cardSetPage') ? 'cardPage' : pageID.slice(0, -4)}Container`)
                            .should('be.visible');
                    })
            })
    });

});
