const test_str = 'test 12345.!???';

beforeEach(() => {
    // Start at `index.html`
    cy.visit('http://localhost:1234/');
})

describe('Add New Card Form', () => {

    // Open the form for each test
    beforeEach(() => {

        // Navigate to card viewer
        cy.get('#cardSetPage').click();
        cy.get('.setContainer')
            .children()
            .first()
            .click();

        // Open the form
        cy.get('button[data-cy="toggle_form"]').click();
    });

    it('Should add a card when both fields are non-empty', () => {

        // Make sure the form is visible and fill it in
        cy.get('form[data-cy="card_form"]')
            .should('be.visible')
            .find('input')
            .each(($elem) => {
                if ($elem.attr('type') != 'submit'){
                    cy.wrap($elem).clear().type(test_str);
                }
            });

        // Submit
        cy.get('input[type="submit"]').click();

        // Check the displayed card
        const cardContainer = cy.get('.cardContainer');
        cardContainer
            .get('.term')
            .find('p')
            .should('have.text', test_str);

        cardContainer
            .get('.description')
            .find('p')
            .should('have.text', test_str);

    });

    it('Should fail when either or both fields are empty', () => {

        // Function to test functionality for arbitrary permutation of the fields
        // Inputs: 
        //  form - Cypress-wrapped form object
        //  ...fields - Variadic list of form field names
        const test_empty_fields = (form, ...fields) => {
            form.find('input')
                .each(($elem) => {

                    if (fields.includes($elem.attr('name'))){
                        // Empty the specified inputs
                        cy.wrap($elem).clear();
                    } else if ($elem.attr('type') != 'submit'){
                        // Fill in the rest
                        cy.wrap($elem).clear().type(test_str);
                    }
                });

            // Submit
            cy.get('input[type="submit"]').click();

            // Check for the error text
            cy.get('p.error').should('be.visible');
        }

        // Test each permutation of empty fields
        test_empty_fields(cy.get('form[data-cy="card_form"]'), 'termInput');
        test_empty_fields(cy.get('form[data-cy="card_form"]'), 'descriptionInput');
        test_empty_fields(cy.get('form[data-cy="card_form"]'), 'termInput', 'descriptionInput');
    });

});

describe('Add New Set Form', () => {

    beforeEach(() => {

        // Navigate to card set page
        cy.get('#cardSetPage').click();

        // Open the form
        cy.get('button[data-cy="toggle_form"]').click();
    });

    it('Should add a card set when the title field is non-empty', () => {

        // Fill in the title field
        cy.get('form[data-cy="set_form"]')
            .find('input')
            .each(($elem) => {
                if ($elem.attr('type') != 'submit'){
                    cy.wrap($elem).clear().type(test_str);
                }
            });

        // Submit
        cy.get('input[type="submit"]').click();

        // Check that the new set is added
        cy.get('.setContainer')
            .children()
            .last()
            .find('li')
            .first()
            .should('have.text', test_str);
    })

    it('Should fail to add a card set when the title field is empty', () => {

        // Make sure the title field is empty - clear it explicitly
        cy.get('form[data-cy="set_form"]')
            .find('input')
            .each(($elem) => {
                if ($elem.attr('type') != 'submit'){
                    cy.wrap($elem).clear();
                }
            });

        // Submit
        cy.get('input[type="submit"]').click();

        // Check for error text
        cy.get('p.error').should('be.visible');
    })

})