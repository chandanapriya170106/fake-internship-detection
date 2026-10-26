// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// Optional: deploy on Remix IDE (Sepolia testnet) to store verified employers
/// and offer-letter hashes on a real blockchain.
contract InternshipRegistry {
    address public admin;
    mapping(address => bool) public verifiedEmployer;
    mapping(bytes32 => address) public offerIssuer; // document hash => employer

    event EmployerVerified(address indexed employer);
    event OfferIssued(bytes32 indexed docHash, address indexed employer);

    constructor() { admin = msg.sender; }

    function verifyEmployer(address employer) external {
        require(msg.sender == admin, "only admin");
        verifiedEmployer[employer] = true;
        emit EmployerVerified(employer);
    }

    function issueOffer(bytes32 docHash) external {
        require(verifiedEmployer[msg.sender], "employer not verified");
        require(offerIssuer[docHash] == address(0), "already issued");
        offerIssuer[docHash] = msg.sender;
        emit OfferIssued(docHash, msg.sender);
    }

    function verifyOffer(bytes32 docHash) external view returns (bool, address) {
        address issuer = offerIssuer[docHash];
        return (issuer != address(0), issuer);
    }
}
