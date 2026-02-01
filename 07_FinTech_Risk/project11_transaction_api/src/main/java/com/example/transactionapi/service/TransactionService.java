package com.example.transactionapi.service;

import com.example.transactionapi.entity.Transaction;
import com.example.transactionapi.entity.User;
import com.example.transactionapi.repository.TransactionRepository;
import com.example.transactionapi.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
@Transactional
public class TransactionService {

    @Autowired
    private TransactionRepository transactionRepository;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private UserService userService;

    public List<Transaction> getAllTransactions() {
        return transactionRepository.findAll();
    }

    public Optional<Transaction> getTransactionById(Long id) {
        return transactionRepository.findById(id);
    }

    public List<Transaction> getTransactionsByUserId(Long userId) {
        return transactionRepository.findByUserIdOrderByTransactionDateDesc(userId);
    }

    public Transaction createTransaction(Transaction transaction) {
        User user = userRepository.findById(transaction.getUser().getId())
                .orElseThrow(() -> new RuntimeException("User not found"));

        transaction.setUser(user);
        transaction.setStatus(Transaction.TransactionStatus.COMPLETED);

        // Update account balance based on transaction type
        updateAccountBalance(transaction);

        return transactionRepository.save(transaction);
    }

    public Transaction updateTransaction(Long id, Transaction transactionDetails) {
        Transaction transaction = transactionRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Transaction not found"));

        transaction.setDescription(transactionDetails.getDescription());
        transaction.setStatus(transactionDetails.getStatus());

        return transactionRepository.save(transaction);
    }

    public void deleteTransaction(Long id) {
        Transaction transaction = transactionRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Transaction not found"));
        transactionRepository.delete(transaction);
    }

    @Transactional
    public Transaction transferMoney(Long senderId, Long recipientId, Double amount, String description) {
        if (amount <= 0) {
            throw new RuntimeException("Transfer amount must be positive");
        }

        User sender = userRepository.findById(senderId)
                .orElseThrow(() -> new RuntimeException("Sender not found"));
        User recipient = userRepository.findById(recipientId)
                .orElseThrow(() -> new RuntimeException("Recipient not found"));

        if (sender.getAccountBalance() < amount) {
            throw new RuntimeException("Insufficient balance");
        }

        // Create transaction for sender (withdrawal)
        Transaction senderTransaction = new Transaction();
        senderTransaction.setTransactionType("TRANSFER_OUT");
        senderTransaction.setAmount(-amount);
        senderTransaction.setDescription(description);
        senderTransaction.setUser(sender);
        senderTransaction.setRecipientAccount(recipient.getUsername());
        senderTransaction.setStatus(Transaction.TransactionStatus.COMPLETED);

        // Create transaction for recipient (deposit)
        Transaction recipientTransaction = new Transaction();
        recipientTransaction.setTransactionType("TRANSFER_IN");
        recipientTransaction.setAmount(amount);
        recipientTransaction.setDescription(description);
        recipientTransaction.setUser(recipient);
        recipientTransaction.setSenderAccount(sender.getUsername());
        recipientTransaction.setStatus(Transaction.TransactionStatus.COMPLETED);

        // Update balances
        sender.setAccountBalance(sender.getAccountBalance() - amount);
        recipient.setAccountBalance(recipient.getAccountBalance() + amount);

        userRepository.save(sender);
        userRepository.save(recipient);

        transactionRepository.save(senderTransaction);
        transactionRepository.save(recipientTransaction);

        return senderTransaction;
    }

    private void updateAccountBalance(Transaction transaction) {
        Double amount = transaction.getAmount();
        User user = transaction.getUser();

        switch (transaction.getTransactionType()) {
            case "DEPOSIT":
                user.setAccountBalance(user.getAccountBalance() + amount);
                break;
            case "WITHDRAWAL":
                if (user.getAccountBalance() < amount) {
                    throw new RuntimeException("Insufficient balance");
                }
                user.setAccountBalance(user.getAccountBalance() - amount);
                break;
            case "TRANSFER_IN":
                user.setAccountBalance(user.getAccountBalance() + amount);
                break;
            case "TRANSFER_OUT":
                if (user.getAccountBalance() < amount) {
                    throw new RuntimeException("Insufficient balance");
                }
                user.setAccountBalance(user.getAccountBalance() - amount);
                break;
        }

        userRepository.save(user);
    }

    public List<Transaction> getTransactionsByDateRange(Long userId, LocalDateTime startDate, LocalDateTime endDate) {
        return transactionRepository.findByUserIdAndDateRange(userId, startDate, endDate);
    }
}