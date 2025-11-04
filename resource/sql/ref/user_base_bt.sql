CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `user_base_bt`
--

DROP TABLE IF EXISTS `user_base_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `user_base_bt` (
  `user_id` varchar(16) NOT NULL COMMENT 'default is NT account, if not, depend on user define(some employee don''t has NT account) -- 2017/11/04',
  `user_name` varchar(32) DEFAULT NULL,
  `user_name_eng` varchar(64) DEFAULT NULL,
  `emp_id` varchar(16) DEFAULT NULL COMMENT 'employee number',
  `role_id` varchar(16) DEFAULT NULL,
  `password` varchar(48) DEFAULT NULL,
  `shift_id` varchar(4) DEFAULT NULL,
  `dept_id` varchar(16) DEFAULT NULL,
  `fab_code` varchar(6) DEFAULT NULL,
  `supervisor_id` varchar(24) DEFAULT NULL,
  `manager_id` varchar(24) DEFAULT NULL,
  `email` varchar(48) DEFAULT NULL,
  `phone` varchar(24) DEFAULT NULL,
  `sex` varchar(6) DEFAULT NULL,
  `active` int(11) DEFAULT NULL COMMENT '0: not 1:active -1:not allow login  -- 2017/11/05 lkchena',
  `rfid_id` varchar(24) DEFAULT NULL COMMENT 'rfid number 2018/05/29 lkchena',
  `recuser` varchar(16) DEFAULT NULL,
  `rectime` datetime DEFAULT NULL,
  PRIMARY KEY (`user_id`),
  UNIQUE KEY `UK_user_base_bt_rfid_id` (`rfid_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AVG_ROW_LENGTH=268 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user_base_bt`
--

LOCK TABLES `user_base_bt` WRITE;
/*!40000 ALTER TABLE `user_base_bt` DISABLE KEYS */;
INSERT INTO `user_base_bt` VALUES ('BKM001','林小明',NULL,NULL,'omi_op','cQGJoZ4ZJecNcuEs',NULL,'製造部',NULL,NULL,NULL,'Tom@goldbkm.com','0987654321','1',NULL,NULL,'BKM001','2020-02-10 16:29:14'),('BKM002','賴先生',NULL,NULL,'omi_qc','cQmqantb2KE?RDiU',NULL,'品管部',NULL,NULL,NULL,'aoo@goldbkm.com','0987654321','1',NULL,NULL,'SYS','2020-03-10 17:11:29'),('BKM003','陳OO',NULL,NULL,'omi_engineer','cQzs.7h:AAFF.WQU',NULL,'研發部',NULL,NULL,NULL,'allew@goldbkm.com','0987654321','1',NULL,NULL,'SYS','2020-02-20 20:10:31'),('SYS','系統管理員',NULL,NULL,'omi_sys','cQ0NwEdQ//saGyVg',NULL,'',NULL,NULL,NULL,'','','1',NULL,NULL,'SYS','2020-02-10 16:29:50'),('TEST','test',NULL,NULL,'omi_rd','cQ/eijIx7Wo=Ui4M',NULL,'',NULL,NULL,NULL,'','','1',NULL,NULL,'SYS','2020-04-24 10:02:01');
/*!40000 ALTER TABLE `user_base_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:29
